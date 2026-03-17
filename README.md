# ⚽ Soccer Video Detection AI Agent

An AI agent for analyzing soccer videos: player detection, team classification, and pitch keypoint detection. Built with YOLO for object detection, OSNet for team re-identification, and HRNet for field keypoint detection.

## Sample Output

<video src="https://github.com/user-attachments/assets/3f62c419-7172-4691-989d-af077f304bbf" controls width="640"></video>

## Features

- **Player detection** — YOLO-based detection of players on the pitch
- **Team classification** — OSNet embeddings + K-means clustering to assign players to Team 1 or Team 2
- **Pitch keypoint detection** — HRNet detects field markers for homography and field normalization
- **Kit color analysis** — Grass-aware color extraction as fallback for team differentiation

## Project Structure

```
soccer-video-detection-ai-agent/
├── src/
│   └── soccer_agent/
│       ├── __init__.py
│       ├── agent.py          # Main AI agent (YOLO + OSNet + HRNet orchestration)
│       └── types.py          # BoundingBox, TVFrameResult models
├── weights/                   # Model weights (git-lfs or download separately)
│   ├── player_detect.pt       # YOLO
│   ├── keypoint_detect.pt     # HRNet
│   ├── osnet_model.pth.tar-100
│   └── hrnetv2_w48.yaml
├── scripts/
│   └── run_video.py           # CLI to process videos
├── run.py                     # Entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Models

### 1. YOLO (`player_detect.pt`)

**Purpose:** Object detection — locates players (and optionally ball, referees) in each video frame.

**Role in the agent:** Runs first on every frame. Outputs bounding boxes with class IDs and confidence scores. Player boxes (class ID 2) are passed to OSNet for team assignment. Also assigns track IDs for temporal consistency across frames.

### 2. OSNet (`osnet_model.pth.tar-100`)

**Purpose:** Person re-identification — produces embedding vectors from cropped upper-body images.

**Role in the agent:** Takes player crops from YOLO boxes and extracts 512‑dim embeddings. Embeddings are aggregated per track, then clustered with K-means (2 clusters) to assign each player to Team 1 or Team 2. Uses kit/jersey appearance to distinguish teams. If OSNet weights are missing, the agent falls back to HSV-based kit color analysis.

### 3. HRNet (`keypoint_detect.pt` + `hrnetv2_w48.yaml`)

**Purpose:** Pitch keypoint detection — detects 32 field markers (lines, corners, markings) in each frame.

**Role in the agent:** Outputs heatmaps for field keypoints. These are mapped to a standard pitch template and refined via homography. Used for field normalization and warping (e.g., bird’s-eye view), not for player body pose.

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Place model weights in `weights/` (see structure above).

3. Run on a video:

```bash
python run.py --video path/to/soccer_video.mp4 --output-dir ./output --save-video
```

Or use the script directly:

```bash
python scripts/run_video.py --video path/to/video.mp4
```

## Contact

Reach out via Telegram: [t.me/whisdev](https://t.me/whisdev)
