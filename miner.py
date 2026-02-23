from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from numpy import ndarray
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gc
import os
import sys
from collections import OrderedDict, defaultdict
from PIL import Image
import torchvision.transforms as T

# ── Grass / kit helpers  ────────────────────────────────

def get_grass_color(img: np.ndarray) -> Tuple[int, int, int]:
    if img is None or img.size == 0:
        return (0, 0, 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    players_imgs, players_boxes = [], []
    for box in result.boxes:
        label = int(box.cls.cpu().numpy()[0])
        if label == 2:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = result.orig_img[y1:y2, x1:x2]
            if crop.size > 0:
                players_imgs.append(crop)
                players_boxes.append((x1, y1, x2, y2))
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)
    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0] // 2, :] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors


# ── OSNet team classification (turbo_7 style)  ────────────────

TEAM_1_ID = 6
TEAM_2_ID = 7
PLAYER_CLS_ID = 2
_OSNET_MODEL = None
osnet_weight_path = None

OSNET_IMAGE_SIZE = (64, 32)  # (height, width)
OSNET_PREPROCESS = T.Compose([
    T.Resize(OSNET_IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _crop_upper_body(frame: ndarray, box: "BoundingBox") -> ndarray:
    return frame[
        max(0, box.y1):max(0, box.y2),
        max(0, box.x1):max(0, box.x2)
    ]


def _preprocess_osnet(crop: ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return OSNET_PREPROCESS(pil)


def _filter_player_boxes(boxes: List["BoundingBox"]) -> List["BoundingBox"]:
    return [b for b in boxes if b.cls_id == PLAYER_CLS_ID]


def _extract_osnet_embeddings(
    frames: List[ndarray],
    batch_boxes: Dict[int, List["BoundingBox"]],
    device: str = "cuda",
) -> Tuple[Optional[ndarray], Optional[List["BoundingBox"]]]:
    global _OSNET_MODEL
    crops = []
    meta = []
    sorted_frame_ids = sorted(batch_boxes.keys())
    for idx, frame_idx in enumerate(sorted_frame_ids):
        frame = frames[idx] if idx < len(frames) else None
        if frame is None:
            continue
        boxes = batch_boxes[frame_idx]
        players = _filter_player_boxes(boxes)
        for box in players:
            crop = _crop_upper_body(frame, box)
            if crop.size == 0:
                continue
            crops.append(_preprocess_osnet(crop))
            meta.append(box)
    if not crops:
        return None, None
    batch = torch.stack(crops).to(device).float()
    with torch.inference_mode():
        embeddings = _OSNET_MODEL(batch)
    del batch
    embeddings = embeddings.cpu().numpy()
    return embeddings, meta


def _aggregate_by_track(
    embeddings: ndarray,
    meta: List["BoundingBox"],
) -> Tuple[ndarray, List["BoundingBox"]]:
    track_map = defaultdict(list)
    box_map = {}
    for emb, box in zip(embeddings, meta):
        key = box.track_id if box.track_id is not None else id(box)
        track_map[key].append(emb)
        box_map[key] = box
    agg_embeddings = []
    agg_boxes = []
    for key, embs in track_map.items():
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-12:
            mean_emb /= norm
        agg_embeddings.append(mean_emb)
        agg_boxes.append(box_map[key])
    return np.array(agg_embeddings), agg_boxes


def _update_team_ids(boxes: List["BoundingBox"], labels: ndarray) -> None:
    for box, label in zip(boxes, labels):
        box.cls_id = TEAM_1_ID if label == 0 else TEAM_2_ID


def _classify_teams_batch(
    frames: List[ndarray],
    batch_boxes: Dict[int, List["BoundingBox"]],
    device: str = "cuda",
) -> None:
    embeddings, meta = _extract_osnet_embeddings(frames, batch_boxes, device)
    if embeddings is None:
        return
    embeddings, agg_boxes = _aggregate_by_track(embeddings, meta)
    n = len(embeddings)
    if n == 0:
        return
    if n == 1:
        agg_boxes[0].cls_id = TEAM_1_ID
        return
    kmeans = KMeans(n_clusters=2, n_init=2, random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    c0, c1 = centroids[0], centroids[1]
    norm_0 = np.linalg.norm(c0)
    norm_1 = np.linalg.norm(c1)
    similarity = np.dot(c0, c1) / (norm_0 * norm_1 + 1e-12)
    if similarity > 0.95:
        for b in agg_boxes:
            b.cls_id = TEAM_1_ID
        return
    if norm_0 <= norm_1:
        kmeans.labels_ = 1 - kmeans.labels_
    _update_team_ids(agg_boxes, kmeans.labels_)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


class LightConvStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        layers = [LightConv3x3(in_channels, out_channels)]
        for _ in range(depth - 1):
            layers.append(LightConv3x3(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1)) if layer_norm else None
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid() if gate_activation == 'sigmoid' else nn.ReLU()

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        return x if self.return_gates else input * x


class OSBlockX1(nn.Module):
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNetX1(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', IN=False):
        super().__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers_list = [block(in_channels, out_channels, IN=IN)]
        for _ in range(1, layer):
            layers_list.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers_list.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers_list)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers_list = []
        for dim in fc_dims:
            layers_list.append(nn.Linear(input_dim, dim))
            layers_list.append(nn.BatchNorm1d(dim))
            layers_list.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers_list.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers_list)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        raise KeyError(f"Unsupported loss: {self.loss}")


def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    return OSNetX1(
        num_classes,
        blocks=[OSBlockX1, OSBlockX1, OSBlockX1],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs,
    )


def load_checkpoint_osnet(fpath):
    fpath = os.path.abspath(os.path.expanduser(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(fpath, map_location=map_location, weights_only=False)
    return checkpoint


def load_pretrained_weights_osnet(model, weight_path):
    checkpoint = load_checkpoint_osnet(weight_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)


def load_osnet(device="cuda", weight_path=None):
    model = osnet_x1_0(num_classes=1, loss='softmax', pretrained=False)
    weight_path = Path(weight_path) if weight_path else None
    if weight_path and weight_path.exists():
        load_pretrained_weights_osnet(model, str(weight_path))
    model.eval()
    model.to(device)
    return model


def _resolve_player_cls_id(model: YOLO, fallback: int = PLAYER_CLS_ID) -> int:
    names = getattr(model, "names", None)
    if not names:
        names = getattr(getattr(model, "model", None), "names", None)
    if isinstance(names, dict):
        for idx, name in names.items():
            if str(name).lower() in ("player", "players"):
                return int(idx)
    if isinstance(names, list):
        for idx, name in enumerate(names):
            if str(name).lower() in ("player", "players"):
                return int(idx)
    return fallback


# ── HRNet architecture  ───────────────────────────────────────────

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False),
                                BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False),
                                BatchNorm2d(num_inchannels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                                          size=[x[i].shape[2], x[i].shape[3]], mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self, config, lines=False, **kwargs):
        self.inplanes = 64
        self.lines = lines
        extra = config['MODEL']['EXTRA']
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        final_inp_channels = sum(pre_stage_channels) + self.inplanes
        self.head = nn.Sequential(nn.Sequential(
            nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=1),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_inp_channels, config['MODEL']['NUM_JOINTS'], kernel_size=extra['FINAL_CONV_KERNEL']),
            nn.Softmax(dim=1) if not self.lines else nn.Sigmoid()))

    def _make_head(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        return self.head(x)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = True if multi_scale_output or i < num_modules - 1 else False
            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels,
                num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x_skip = x.clone()
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            x_list.append(self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            x_list.append(self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i])
        x = self.stage4(x_list)

        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        return self._make_head(x, x_skip)

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrained:
            if os.path.isfile(pretrained):
                pretrained_dict = torch.load(pretrained)
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
            else:
                sys.exit(f'Weights {pretrained} not found.')

def get_cls_net(config, pretrained='', **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights(pretrained)
    return model


# ── Keypoint mapping & inference helpers  ─────────────────────────

map_keypoints = {
    1: 1, 2: 14, 3: 25, 4: 2, 5: 10, 6: 18, 7: 26, 8: 3, 9: 7, 10: 23,
    11: 27, 20: 4, 21: 8, 22: 24, 23: 28, 24: 5, 25: 13, 26: 21, 27: 29,
    28: 6, 29: 17, 30: 30, 31: 11, 32: 15, 33: 19, 34: 12, 35: 16, 36: 20,
    45: 9, 50: 31, 52: 32, 57: 22
}

# Template keypoints for homography refinement (new-5 style)
TEMPLATE_F0: List[Tuple[float, float]] = [
    (5, 5), (5, 140), (5, 250), (5, 430), (5, 540), (5, 675), (55, 250), (55, 430),
    (110, 340), (165, 140), (165, 270), (165, 410), (165, 540), (527, 5), (527, 253),
    (527, 433), (527, 675), (888, 140), (888, 270), (888, 410), (888, 540), (940, 340),
    (998, 250), (998, 430), (1045, 5), (1045, 140), (1045, 250), (1045, 430), (1045, 540),
    (1045, 675), (435, 340), (615, 340),
]
TEMPLATE_F1: List[Tuple[float, float]] = [
    (2.5, 2.5), (2.5, 139.5), (2.5, 249.5), (2.5, 430.5), (2.5, 540.5), (2.5, 678),
    (54.5, 249.5), (54.5, 430.5), (110.5, 340.5), (164.5, 139.5), (164.5, 269), (164.5, 411),
    (164.5, 540.5), (525, 2.5), (525, 249.5), (525, 430.5), (525, 678), (886.5, 139.5),
    (886.5, 269), (886.5, 411), (886.5, 540.5), (940.5, 340.5), (998, 249.5), (998, 430.5),
    (1048, 2.5), (1048, 139.5), (1048, 249.5), (1048, 430.5), (1048, 540.5), (1048, 678),
    (434.5, 340), (615.5, 340),
]
HOMOGRAPHY_FILL_ONLY_VALID = True
KP_THRESHOLD = 0.2  # new-5 style (was 0.3)
HRNET_BATCH_SIZE = 4  # larger batch = faster (if GPU mem allows)


def _preprocess_batch(frames):
    target_h, target_w = 540, 960
    batch = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h)).astype(np.float32) / 255.0
        batch.append(np.transpose(img, (2, 0, 1)))
    return torch.from_numpy(np.stack(batch)).float()


def _extract_keypoints(heatmap: torch.Tensor, scale: int = 2):
    b, c, h, w = heatmap.shape
    max_pooled = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    local_maxima = (max_pooled == heatmap)
    masked = heatmap * local_maxima
    flat = masked.view(b, c, -1)
    scores, indices = torch.topk(flat, 1, dim=-1, sorted=False)
    y_coords = torch.div(indices, w, rounding_mode="floor") * scale
    x_coords = (indices % w) * scale
    return torch.stack([x_coords.float(), y_coords.float(), scores], dim=-1)


def _process_keypoints(kp_coords, threshold, w, h, batch_size):
    kp_np = kp_coords.cpu().numpy()
    results = []
    for b_idx in range(batch_size):
        kp_dict = {}
        valid = np.where(kp_np[b_idx, :, 0, 2] > threshold)[0]
        for ch_idx in valid:
            kp_dict[ch_idx + 1] = {
                'x': float(kp_np[b_idx, ch_idx, 0, 0]) / w,
                'y': float(kp_np[b_idx, ch_idx, 0, 1]) / h,
                'p': float(kp_np[b_idx, ch_idx, 0, 2]),
            }
        results.append(kp_dict)
    return results


def _run_hrnet_batch(frames, model, threshold, batch_size=8):
    if not frames or model is None:
        return []
    device = next(model.parameters()).device
    results = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        batch = _preprocess_batch(chunk).to(device)
        with torch.no_grad():
            heatmaps = model(batch)
        kp_coords = _extract_keypoints(heatmaps[:, :-1, :, :], scale=2)
        batch_kps = _process_keypoints(kp_coords, threshold, 960, 540, len(chunk))
        results.extend(batch_kps)
        del heatmaps, kp_coords, batch
        gc.collect()
    return results


def _apply_keypoint_mapping(kp_dict):
    return {map_keypoints[k]: v for k, v in kp_dict.items() if k in map_keypoints}


def _normalize_keypoints(kp_results, frames, n_keypoints):
    keypoints = []
    max_frames = min(len(kp_results), len(frames))
    for i in range(max_frames):
        kp_dict = kp_results[i]
        h, w = frames[i].shape[:2]
        frame_kps = []
        for idx in range(n_keypoints):
            kp_idx = idx + 1
            x, y = 0, 0
            if kp_idx in kp_dict:
                d = kp_dict[kp_idx]
                if isinstance(d, dict) and 'x' in d:
                    x = int(d['x'] * w)
                    y = int(d['y'] * h)
            frame_kps.append((x, y))
        keypoints.append(frame_kps)
    return keypoints


def _fix_keypoints(kps: list, n: int) -> list:
    if len(kps) < n:
        kps += [(0, 0)] * (n - len(kps))
    elif len(kps) > n:
        kps = kps[:n]

    if kps[2] != (0,0) and kps[4] != (0,0) and kps[3] == (0,0):
        kps[3] = kps[4]; kps[4] = (0,0)
    if kps[0] != (0,0) and kps[4] != (0,0) and kps[1] == (0,0):
        kps[1] = kps[4]; kps[4] = (0,0)
    if kps[2] != (0,0) and kps[3] != (0,0) and kps[1] == (0,0) and kps[3][0] > kps[2][0]:
        kps[1] = kps[3]; kps[3] = (0,0)
    if kps[28] != (0,0) and kps[25] == (0,0) and kps[26] != (0,0) and kps[26][0] > kps[28][0]:
        kps[25] = kps[28]; kps[28] = (0,0)
    if kps[24] != (0,0) and kps[28] != (0,0) and kps[25] == (0,0):
        kps[25] = kps[28]; kps[28] = (0,0)
    if kps[24] != (0,0) and kps[27] != (0,0) and kps[26] == (0,0):
        kps[26] = kps[27]; kps[27] = (0,0)
    if kps[28] != (0,0) and kps[23] == (0,0) and kps[20] != (0,0) and kps[20][1] > kps[23][1]:
        kps[23] = kps[20]; kps[20] = (0,0)
    return kps


def _keypoints_to_float(keypoints: list) -> List[List[float]]:
    """Convert keypoints to [[x, y], ...] float format for homography."""
    return [[float(x), float(y)] for x, y in keypoints]


def _keypoints_to_int(keypoints: list) -> List[Tuple[int, int]]:
    """Convert keypoints to [(x, y), ...] integer format."""
    return [(int(round(float(kp[0]))), int(round(float(kp[1])))) for kp in keypoints]


def _apply_homography_refinement(
    keypoints: List[List[float]],
    frame: np.ndarray,
    n_keypoints: int,
) -> List[List[float]]:
    """Refine keypoints using homography from template to frame (new-5 style)."""
    if n_keypoints != 32 or len(TEMPLATE_F0) != 32 or len(TEMPLATE_F1) != 32:
        return keypoints
    frame_height, frame_width = frame.shape[:2]
    valid_src: List[Tuple[float, float]] = []
    valid_dst: List[Tuple[float, float]] = []
    valid_indices: List[int] = []
    for kp_idx, kp in enumerate(keypoints):
        if kp and len(kp) >= 2:
            x, y = float(kp[0]), float(kp[1])
            if not (abs(x) < 1e-6 and abs(y) < 1e-6) and 0 <= x < frame_width and 0 <= y < frame_height:
                valid_src.append(TEMPLATE_F1[kp_idx])
                valid_dst.append((x, y))
                valid_indices.append(kp_idx)
    if len(valid_src) < 4:
        return keypoints
    src_pts = np.array(valid_src, dtype=np.float32)
    dst_pts = np.array(valid_dst, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return keypoints
    all_template_points = np.array(TEMPLATE_F0, dtype=np.float32).reshape(-1, 1, 2)
    adjusted_points = cv2.perspectiveTransform(all_template_points, H)
    adjusted_points = adjusted_points.reshape(-1, 2)
    adj_x = adjusted_points[:32, 0]
    adj_y = adjusted_points[:32, 1]
    valid_mask = (adj_x >= 0) & (adj_y >= 0) & (adj_x < frame_width) & (adj_y < frame_height)
    valid_indices_set = set(valid_indices)
    adjusted_kps: List[List[float]] = [[0.0, 0.0] for _ in range(32)]
    for i in np.where(valid_mask)[0]:
        if not HOMOGRAPHY_FILL_ONLY_VALID or i in valid_indices_set:
            adjusted_kps[i] = [float(adj_x[i]), float(adj_y[i])]
    return adjusted_kps


# ── Pydantic models ───────────────────────────────────────────────────────────

# Team assignment: 6 = team 1, 7 = team 2
TEAM_1_ID = 6
TEAM_2_ID = 7
PLAYER_CLS_ID = 2


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float
    track_id: Optional[int] = None

class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: List[Tuple[float, float]]  # [(x, y), ...] float coordinates


# ── Miner ─────────────────────────────────────────────────────────────────────

class Miner:
    def __init__(self, path_hf_repo: Path) -> None:
        self.path_hf_repo = path_hf_repo
        self.is_start = False
        self._executor = ThreadPoolExecutor(max_workers=2)

        global _OSNET_MODEL, osnet_weight_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # BBox model
        bbox_file = "player_detect.pt"
        self.bbox_model = YOLO(Path(bbox_file) if Path(bbox_file).exists() else path_hf_repo / bbox_file)
        print("✅ BBox Model Loaded")
        global PLAYER_CLS_ID
        PLAYER_CLS_ID = _resolve_player_cls_id(self.bbox_model, PLAYER_CLS_ID)

        # OSNet team classifier
        osnet_weight_path = path_hf_repo / "osnet_model.pth.tar-100"
        if osnet_weight_path.exists():
            _OSNET_MODEL = load_osnet(device, osnet_weight_path)
            print("✅ Team Classifier Loaded (OSNet)")
        else:
            _OSNET_MODEL = None
            print(f"⚠️ OSNet weights not found at {osnet_weight_path}. Using HSV fallback.")

        # Keypoints model: HRNet 
        kp_config_file  = "hrnetv2_w48.yaml"
        kp_weights_file = "keypoint_detect.pt"
        config_path  = Path(kp_config_file)  if Path(kp_config_file).exists()  else path_hf_repo / kp_config_file
        weights_path = Path(kp_weights_file) if Path(kp_weights_file).exists() else path_hf_repo / kp_weights_file
        cfg = yaml.safe_load(open(config_path, 'r'))
        hrnet = get_cls_net(cfg)
        state = torch.load(weights_path, map_location=device, weights_only=False)
        hrnet.load_state_dict(state)
        hrnet.to(device).eval()
        self.keypoints_model = hrnet
        print("✅ HRNet Keypoints Model Loaded")

    def __repr__(self) -> str:
        return (
            f"BBox Model: {type(self.bbox_model).__name__}\n"
            f"Keypoints Model: {type(self.keypoints_model).__name__}\n"
            f"Team Clustering: OSNet + KMeans"
        )

    def _bbox_task(self, images: list[ndarray]) -> list[list[BoundingBox]]:
        """Batch YOLO inference + team assignment."""
        if not images:
            return []
        if self.bbox_model is None:
            return [[] for _ in images]
        try:
            bbox_results = self.bbox_model(images, conf=0.2, iou=0.5, agnostic_nms=True, verbose=False)
        except Exception:
            return [[] for _ in images]
        bboxes_by_frame: Dict[int, List[BoundingBox]] = {}
        track_id = 0
        for frame_idx, bbox_result in enumerate(bbox_results):
            boxes = []
            if bbox_result and bbox_result.boxes is not None and len(bbox_result.boxes) > 0:
                for box in bbox_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf.cpu().numpy()[0])
                    cls_id = int(box.cls.cpu().numpy()[0])
                    tid = None
                    if cls_id == PLAYER_CLS_ID:
                        track_id += 1
                        tid = track_id
                    boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, cls_id=cls_id, conf=conf, track_id=tid))
            bboxes_by_frame[frame_idx] = boxes

        try:
            _classify_teams_batch(images, bboxes_by_frame, self.device)
        except Exception:
            pass
        return [bboxes_by_frame[i] for i in range(len(images))]

    def _keypoint_task(self, images: list[ndarray], n_keypoints: int) -> list[list]:
        """HRNet keypoints + homography refinement."""
        if not images:
            return []
        if self.keypoints_model is None:
            return [[(0, 0)] * n_keypoints for _ in images]
        try:
            raw_kps = _run_hrnet_batch(images, self.keypoints_model, KP_THRESHOLD, batch_size=HRNET_BATCH_SIZE)
        except Exception:
            return [[(0, 0)] * n_keypoints for _ in images]
        raw_kps = [_apply_keypoint_mapping(kp) for kp in raw_kps] if raw_kps else []
        keypoints = _normalize_keypoints(raw_kps, images, n_keypoints) if raw_kps else [[(0, 0)] * n_keypoints for _ in images]
        keypoints = [_fix_keypoints(kps, n_keypoints) for kps in keypoints]
        keypoints = [_keypoints_to_float(kps) for kps in keypoints]
        if n_keypoints == 32 and len(TEMPLATE_F0) == 32 and len(TEMPLATE_F1) == 32:
            for idx in range(len(images)):
                try:
                    keypoints[idx] = _apply_homography_refinement(keypoints[idx], images[idx], n_keypoints)
                except Exception:
                    pass
        # keypoints = [_keypoints_to_int(kps) for kps in keypoints]
        return keypoints

    def predict_batch(
        self,
        batch_images: list[ndarray],
        offset: int,
        n_keypoints: int,
    ) -> list[TVFrameResult]:

        if not self.is_start:
            self.is_start = True

        images = list(batch_images)
        if offset == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Run bbox (batched YOLO) and keypoints in parallel
        future_bbox = self._executor.submit(self._bbox_task, images)
        future_kp = self._executor.submit(self._keypoint_task, images, n_keypoints)
        bbox_per_frame = future_bbox.result()
        keypoints = future_kp.result()

        return [
            TVFrameResult(frame_id=offset + i, boxes=bbox_per_frame[i], keypoints=keypoints[i])
            for i in range(len(images))
        ]