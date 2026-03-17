#!/usr/bin/env python3
"""
Run soccer video detection on video files.

Processes video(s) with the AI agent: player detection, team classification,
and pitch keypoint detection. Outputs annotated video and JSON results.
"""

from typing import List
from pathlib import Path

import numpy as np
import argparse
import json
import re
import sys
import cv2

# Project root and src for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from soccer_agent import AiAgent, TVFrameResult


def load_frames(video_path: Path, max_frames: int = None, start_frame: int = 0) -> List:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is None or max_frames <= 0:
        max_frames = total_frames - start_frame

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def _infer_so_py_tag(so_path: Path) -> str | None:
    """Extract CPython ABI tag like 'cp312' from a .so filename."""
    match = re.search(r"cpython-(\d)(\d+)", so_path.name)
    if not match:
        return None
    major = match.group(1)
    minor = match.group(2)
    return f"cp{major}{minor}"


def _current_py_tag() -> str:
    """Return current CPython tag like 'cp311'."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def visualize_results(
    frame: np.ndarray,
    result: TVFrameResult,
    show_boxes: bool = True,
    show_keypoints: bool = True,
    show_warped_template: bool = True,
    template_alpha: float = 0.3,
    return_warped_template: bool = False
) -> np.ndarray:
    """Visualize detection results on a frame."""
    vis_frame = frame.copy()
    warped_template_output = None

    if show_boxes and result.boxes:
        for box in result.boxes:
            colors = {
                0: (0, 255, 255), 1: (255, 0, 255), 3: (255, 255, 0),
                6: (0, 0, 255), 7: (0, 255, 0), 2: (255, 255, 255),
            }
            label_names = {0: "Ball", 1: "Goalkeeper", 3: "Referee", 6: "Team1", 7: "Team2", 2: "Player"}

            if box.cls_id == 6:
                color, label_name = (0, 0, 255), "Team1"
            elif box.cls_id == 7:
                color, label_name = (0, 255, 0), "Team2"
            elif box.cls_id == 2:
                team_id = getattr(box, 'team_id', None) or getattr(box, 'team', None)
                if team_id:
                    team_str = str(team_id).strip().lower()
                    if team_str in {"1", "team1"}:
                        color, label_name = (0, 0, 255), "Team1"
                    elif team_str in {"2", "team2"}:
                        color, label_name = (0, 255, 0), "Team2"
                    else:
                        color, label_name = (255, 255, 255), "Player"
                else:
                    color = colors.get(box.cls_id, (255, 255, 255))
                    label_name = label_names.get(box.cls_id, "Player")
            else:
                color = colors.get(box.cls_id, (255, 255, 255))
                label_name = label_names.get(box.cls_id, f"C{box.cls_id}")

            cv2.rectangle(vis_frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            cv2.putText(vis_frame, f"{label_name}:{box.conf:.2f}", (box.x1, box.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    idx_list = []
    if show_keypoints and result.keypoints:
        for idx, (x, y) in enumerate(result.keypoints):
            if (x, y) != (0, 0):
                x, y = int(x), int(y)
                cv2.circle(vis_frame, (x, y), 6, (0, 255, 255), -1)
                cv2.circle(vis_frame, (x, y), 8, (255, 255, 255), 2)
                idx_list.append(idx)
                cv2.putText(vis_frame, str(idx + 1), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    info_text = f"Frame {result.frame_id} | Boxes:{len(result.boxes)} | KPs:{sum(1 for kp in result.keypoints if kp != (0, 0))}/32, {idx_list}"
    cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if return_warped_template:
        return vis_frame, warped_template_output
    return vis_frame


def format_results_as_chute_output(results: List[TVFrameResult]) -> dict:
    """Format results to match chute template output format."""
    try:
        frame_results = [frame_result.model_dump() for frame_result in results]
        return {"success": True, "predictions": {"frames": frame_results}, "error": None}
    except Exception as e:
        return {"success": False, "predictions": None, "error": str(e)}


def save_results(
    frames: List[np.ndarray],
    results: List[TVFrameResult],
    output_dir: Path,
    output_filename: str = "output_video.mp4",
    save_video: bool = True,
    save_json: bool = True,
    fps: float = 25.0,
    show_warped_template: bool = True,
    template_alpha: float = 0.3,
    save_warped_templates: bool = True
):
    """Save visualization results."""
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}\nOutput directory: {output_dir}")

    if save_json and results:
        json_path = output_dir / f"{Path(output_filename).stem}_results.json"
        formatted_output = format_results_as_chute_output(results)
        with open(json_path, 'w') as f:
            json.dump(formatted_output, f, indent=2)
        print(f"✅ JSON results saved: {json_path}")

    vis_frames = []
    warped_templates = []
    if save_warped_templates and show_warped_template:
        warped_dir = output_dir / f"{Path(output_filename).stem}_warped_templates"
        warped_dir.mkdir(exist_ok=True)

    for frame, result in zip(frames, results):
        vis_output = visualize_results(
            frame, result,
            show_warped_template=show_warped_template,
            template_alpha=template_alpha,
            return_warped_template=save_warped_templates and show_warped_template
        )
        if isinstance(vis_output, tuple):
            vis_frame, warped_template = vis_output
            vis_frames.append(vis_frame)
            if warped_template is not None:
                warped_templates.append(warped_template)
                warped_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(warped_dir / f"frame_{result.frame_id:06d}_warped.png"), warped_template)
        else:
            vis_frames.append(vis_output)

    if save_video and vis_frames:
        video_path = output_dir / output_filename
        h, w = vis_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        for vis_frame in vis_frames:
            video_writer.write(vis_frame)
        video_writer.release()
        print(f"✅ Video saved: {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer Video Detection AI Agent")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory containing video files")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames (0 = all)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--offset", type=int, default=10, help="Frame offset for predict_batch")
    parser.add_argument("--n-keypoints", type=int, default=32, help="Number of keypoints")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--show-warped-template", action="store_true", default=True)
    parser.add_argument("--template-alpha", type=float, default=0.3)
    parser.add_argument("--save-warped-templates", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_paths = [video_path]
    else:
        video_dir = PROJECT_ROOT / args.video_dir
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        video_paths = sorted(video_dir.glob("*.mp4"))
        if not video_paths:
            raise FileNotFoundError(f"No .mp4 files in: {video_dir}")

    ai_agent = AiAgent(PROJECT_ROOT)

    print(f"{'='*70}\nSoccer Video Detection AI Agent\n{'='*70}")
    print(f"Videos: {len(video_paths)}, Output: {output_dir}\n")

    for video_idx, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*70}\nVIDEO {video_idx}/{len(video_paths)}: {video_path.name}\n{'='*70}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        max_frames = args.frames if args.frames > 0 else total_frames - args.start_frame

        if video_idx == 1:
            warmup_frames = load_frames(video_path, min(2, max_frames), args.start_frame)
            ai_agent.predict_batch(warmup_frames, args.offset, args.n_keypoints)

        all_frames, all_results = [], []
        num_batches = (max_frames + args.batch_size - 1) // args.batch_size

        for batch_number in range(num_batches):
            batch_start = args.start_frame + batch_number * args.batch_size
            batch_size = min(args.batch_size, max_frames - batch_number * args.batch_size)
            frame_number = args.start_frame + batch_number * args.batch_size

            print(f"Predicting Batch: {batch_number + 1}/{num_batches}")
            batch_frames = load_frames(video_path, batch_size, batch_start)
            if not batch_frames:
                break

            batch_results = ai_agent.predict_batch(
                batch_images=batch_frames,
                offset=frame_number,
                n_keypoints=args.n_keypoints,
            )
            if batch_results:
                all_frames.extend(batch_frames)
                all_results.extend(batch_results)

        print(f"\nProcessed: {len(all_results)} frames")
        if all_results and (args.save_video or args.save_json):
            save_results(
                all_frames, all_results, output_dir,
                output_filename=video_path.name,
                save_video=args.save_video, save_json=args.save_json,
                fps=fps,
                show_warped_template=args.show_warped_template,
                template_alpha=args.template_alpha,
                save_warped_templates=args.save_warped_templates
            )

    print(f"\n{'='*70}\n✅ ALL {len(video_paths)} VIDEOS PROCESSED!\n{'='*70}")


if __name__ == "__main__":
    main()
