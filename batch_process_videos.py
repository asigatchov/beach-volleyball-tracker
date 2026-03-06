# batch_process_videos.py
import os
import subprocess
import argparse
import sys
import shlex
from datetime import datetime

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

def print_external_command(command):
    print(f"  [Cmd] {shlex.join(command)}")

def main():
    parser = argparse.ArgumentParser(description="[Final Release] Automated batch volleyball video analysis pipeline.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing multiple video files.")
    parser.add_argument("--output_folder", type=str, default="batch_processing_results", help="Root output folder to store all results.")
    parser.add_argument("--hit_dist", type=float, default=320, help="[Analysis Param] Maximum contact distance at hit moment.")
    parser.add_argument("--save_annotated_frames", action="store_true", help="[Debug] Save every annotated frame.")
    parser.add_argument("--save_original_frames", action="store_true", help="[Training] Save every original (unannotated) frame.")
    args = parser.parse_args()

    script_start_time = datetime.now()
    print(f"--- Batch processing started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    video_files = find_video_files(args.input_folder)
    if not video_files:
        print(f"[Error] No videos found in folder '{args.input_folder}'."); return

    total_videos = len(video_files)
    print(f"[Info] Found {total_videos} video(s). Starting...")

    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for idx, video_path in enumerate(video_files):
        video_start_time = datetime.now()
        print(f"\n--- [{idx + 1}/{total_videos}] Processing video: {os.path.basename(video_path)} ---")
        
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        os.makedirs(video_specific_output_dir, exist_ok=True)
        log_file_path = os.path.join(video_specific_output_dir, f"{video_base_name}_processing_log.log")
        
        # --- Stage 1: Tracking ---
        print("  [Step 1/2] Running object and pose tracking... (logs will be written to file)")
        tracking_output_path = os.path.join(video_specific_output_dir, 'tracking_output')
        track_command = [
            sys.executable, os.path.join("video_processing", "track_ball_and_player.py"),
            "--input", video_path,
            "--output_dir", tracking_output_path,
            "--log_file", log_file_path,
            "--log_mode", "w"
        ]
        if args.save_annotated_frames: track_command.append("--save_annotated_frames")
        if args.save_original_frames: track_command.append("--save_original_frames")

        print_external_command(track_command)
        result_track = subprocess.run(track_command)

        if result_track.returncode != 0:
            print(f"  [Error] Tracking stage failed. See log file for details: {log_file_path}"); continue
        print("  [Done] Tracking succeeded.")

        # --- Stage 2: Analysis ---
        print("  [Step 2/2] Running serve event analysis... (logs will be written to file)")
        analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
        json_input_path = os.path.join(tracking_output_path, video_base_name, f"{video_base_name}_all_frames_data_with_pose.json")

        if not os.path.exists(json_input_path):
             print(f"  [Error] JSON file not found. Unable to run analysis."); continue

        analyze_command = [
            sys.executable, os.path.join("video_processing", "test_serve_analyzer.py"),
            "--video_input", video_path,
            "--json_input", json_input_path,
            "--output_dir", analysis_output_path,
            "--hit_dist", str(args.hit_dist),
            "--log_file", log_file_path,
            "--log_mode", "a"
        ]
        print_external_command(analyze_command)
        result_analyze = subprocess.run(analyze_command)

        if result_analyze.returncode != 0:
            print(f"  [Error] Analysis stage failed. See log file for details: {log_file_path}"); continue
        print("  [Done] Analysis succeeded.")
        print(f"  [Log] Full processing record saved to: {log_file_path}")

    print(f"\n--- All videos processed. Total elapsed time: {datetime.now() - script_start_time} ---")

if __name__ == '__main__':
    main()
