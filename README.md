# Beach Volleyball Advanced Tactical Analysis Project

## Project Overview

This project provides an automated solution for extracting serve-related tactical data from beach volleyball match videos. Using advanced computer vision techniques (YOLOv8 object tracking and pose estimation), the system not only detects serve events accurately, but also quantifies serve **type** and **landing zone**.

The core value of this tool is its **stability** and **tuneability**. After multiple iterations, the current algorithm is based on physical intuition and validation against real-world scenarios, making it reliable for coaches, players, and data analysts.

---

## Core Features

- **High-accuracy serve detection**: Uses a validated and stable four-stage state machine (`search toss` -> `confirm toss` -> `wait apex` -> `wait hit`) to reduce false positives caused by gravity-driven drops or player movement.

- **Precise server identification**:
  - **Backtracking method**: After an initial hit detection, the system rewinds a few frames (`--search_offset`) and identifies the server at the best moment when player-ball distance is minimal.
  - **Point-to-box shortest distance algorithm**: Reduces perspective-related errors by measuring the shortest distance from a point to a rectangle edge, improving player-ball association.

- **Advanced serve type analysis (jump serve / standing serve)**:
  - **Pose trajectory analysis**: Extracts body keypoints with `yolov8n-pose.pt`.
  - **Single-window displacement method**: Uses hip vertical displacement before the serve and computes the distance from lowest crouch to highest jump point to classify serve type robustly, even with occasional per-frame detection failures.

- **Serve landing zone analysis (A/B/C)**:
  - Requires a `court_config.json` file that defines the four court corners.
  - The program automatically divides the baseline into three regions and classifies landing zone as A, B, or C based on server position.

- **Strong visual debugging tools**:
  - **Jump-serve trajectory plot**: With `--debug_jump_serve`, the program generates a hip-height trajectory chart for each jump-serve decision.
  - **Key-frame saving**: Automatically saves three frames before and after each detected serve event for fast manual review.

- **Automated data export**:
  - After analysis, results from all videos (segment name, serve zone, serve type) are aggregated into `analysis_summary.csv`, ready for Excel or other analysis tools.

- **Controllable parallel processing**:
  - Use `--workers` to set concurrent video processing count and balance execution speed vs hardware stability to avoid memory exhaustion.

---

## Installation and Requirements

1. **Python environment**: Python 3.8+ is recommended.

2. **Conda (recommended)**:
   ```bash
   conda create -n beach-volleyball-tracker python=3.8
   conda activate beach-volleyball-tracker
   ```

3. **Dependencies**:
   ```bash
   pip install opencv-python numpy tqdm ultralytics matplotlib
   ```
   *(The `ultralytics` package installs `torch` and related dependencies automatically.)*

4. **Model files**:
   - Ensure `video_processing/track_ball_and_player.py` can access your selected model files, e.g. `yolov8s-pose.pt` and `ball_best.pt`.
   - If using the latest provided tracking script, ensure `yolov8n.pt` and `yolov8n-pose.pt` are available locally or can be auto-downloaded.

---

## Usage Guide

The workflow has two main steps:

### Step 1: Define Court Boundaries (once per camera/view type)

To enable serve zone analysis, create a config file with the four court corner coordinates.

1. **Run the config generator**:
   ```bash
   python court_definition/court_config_generator.py --video_path "path/to/your/sample_video.mp4"
   ```

2. **Mark court corners**:
   - The program displays the first frame.
   - Click the four corners strictly in this order: `top-left -> bottom-left -> bottom-right -> top-right`.
   - After completion, `court_config.json` is saved in the project root.

### Step 2: Run the main analysis script

This is the core step for analyzing all videos and generating outputs.

1. **Basic command**:
   ```bash
   python run_analysis_all_in_one.py --input_folder "your_video_folder" --court_config "court_config.json"
   ```

2. **Full debug mode (recommended for first runs or parameter tuning)**:
   ```bash
   python run_analysis_all_in_one.py --input_folder "your_video_folder" --court_config "court_config.json" --workers 1 --debug_jump_serve --overwrite
   ```
   This enables jump-serve debug plots and forces a single worker for stability.

---

## Command-line Parameters

Main script `run_analysis_all_in_one.py` exposes rich parameters for customization.

### Main Parameters

| Parameter | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input_folder` | **Yes** | - | Path to folder containing videos to analyze. |
| `--court_config` | No | `None` | Path to `court_config.json`. Enables serve zone analysis when provided. |
| `--workers` | No | `2` | Number of videos processed in parallel. Tune based on RAM/VRAM. |
| `--overwrite` | No | `False` | Re-analyze all videos and overwrite old results. |
| `--debug_jump_serve` | No | `False` | Enable jump-serve visualization debug mode (hip trajectory plots). |

### Core Algorithm Parameters (v11 stable)

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--hit_v` | `40.0` | Minimum instantaneous total speed to detect a hit. |
| `--toss_vy` | `8.0` | Minimum initial upward vertical speed to trigger possible toss. |
| `--vertical_ratio` | `1.5` | During toss, vertical speed must be at least this multiple of horizontal speed. |
| `--hit_h_ratio` | `2.5` | **Important**: Maximum horizontal/vertical speed ratio at hit time, used to filter pure vertical drops. |
| *(... other state-machine params such as `max_frames_to_apex` ...)* | ... | Parameters for tuning state-machine time windows. |

### Advanced Analysis Parameters

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--search_offset` | `3` | Number of frames to backtrack from initial hit point when searching for server. |
| `--ball_leave_threshold` | `30` | **Deprecated**: Parameter used by old fine timing logic. |
| `--jump_height_threshold` | `20` | Minimum hip vertical displacement (pixels) to classify as a jump. |

---

## Output Description

After analysis, all outputs are stored under your specified `--output_folder`.

- **`final_summary_refined/`**: Contains final analysis results.
  - **`analysis_summary.csv`**: Most important output; table for Excel containing serve zone and serve type per segment.
  - **`summary_report.txt`**: Text summary report including debug status.
  - **`..._jump_debug_plot.png`**: (Debug mode only) Hip-height trajectory plot for jump-serve validation.

- **`key_frames_for_review_refined/`**: Stores key frames for **all videos** (3 frames before and after `HIT`) for quick manual review.

- **`[video_name]/`**: Per-video folder.
  - `tracking_output/`: Raw object-tracking data (`..._all_frames_data_with_pose.json`).
  - `key_frames/`: Key frames for this specific video.

---

## Troubleshooting

- **Inaccurate serve detection (falling ball misclassified as hit)**:
  - This is a key issue addressed by `v11`. Try lowering `--hit_h_ratio` (for example from `2.5` to `1.5` or `1.0`) so a hit must include stronger horizontal motion.

- **Incorrect jump serve / standing serve classification**:
  1. Re-run with `--debug_jump_serve`.
  2. Inspect generated `..._jump_debug_plot.png`.
  3. Check `Displacement` in the plot title. If an obvious jump serve has displacement slightly below threshold (default `20`, e.g. `18.5`), lower `--jump_height_threshold`.

- **Program crashes with memory errors (`CUDA error` or `MemoryError`)**:
  - Too many videos are being processed in parallel. Reduce `--workers`, starting from `--workers 1`.
