# batch_process_videos.py
import os
import subprocess
import argparse
import sys
from datetime import datetime

def find_video_files(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                video_files.append(os.path.join(root, file))
    return video_files

def main():
    parser = argparse.ArgumentParser(description="【最終交付版】自動化批次處理排球影片分析系統。")
    parser.add_argument("--input_folder", type=str, required=True, help="包含多個影片檔案的輸入資料夾。")
    parser.add_argument("--output_folder", type=str, default="batch_processing_results", help="存放所有結果的總輸出資料夾。")
    parser.add_argument("--hit_dist", type=float, default=320, help="[分析參數] 擊球時的最大接觸距離。")
    parser.add_argument("--save_annotated_frames", action="store_true", help="[偵錯] 儲存「已標註」的每一幀畫面。")
    parser.add_argument("--save_original_frames", action="store_true", help="[訓練] 儲存「未標註」的原始每一幀畫面。")
    args = parser.parse_args()

    script_start_time = datetime.now()
    print(f"--- 批次處理開始於: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    video_files = find_video_files(args.input_folder)
    if not video_files:
        print(f"[錯誤] 在資料夾 '{args.input_folder}' 中找不到任何影片。"); return

    total_videos = len(video_files)
    print(f"[資訊] 找到 {total_videos} 個影片，準備開始...")

    base_output_dir = os.path.abspath(args.output_folder)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for idx, video_path in enumerate(video_files):
        video_start_time = datetime.now()
        print(f"\n--- [{idx + 1}/{total_videos}] 開始處理影片: {os.path.basename(video_path)} ---")
        
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_specific_output_dir = os.path.join(base_output_dir, video_base_name)
        os.makedirs(video_specific_output_dir, exist_ok=True)
        log_file_path = os.path.join(video_specific_output_dir, f"{video_base_name}_處理日誌.log")
        
        # --- 階段 1: 追蹤 ---
        print("  [步驟 1/2] 正在執行物件與姿態追蹤... (日誌將直接寫入檔案)")
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
        
        result_track = subprocess.run(track_command)

        if result_track.returncode != 0:
            print(f"  [錯誤] 追蹤階段失敗！詳情請見日誌檔案: {log_file_path}"); continue
        print("  [完成] 追蹤成功！")

        # --- 階段 2: 分析 ---
        print("  [步驟 2/2] 正在執行發球事件分析... (日誌將直接寫入檔案)")
        analysis_output_path = os.path.join(video_specific_output_dir, 'analysis_output')
        json_input_path = os.path.join(tracking_output_path, video_base_name, f"{video_base_name}_all_frames_data_with_pose.json")

        if not os.path.exists(json_input_path):
             print(f"  [錯誤] 找不到 JSON 檔案，無法分析。"); continue

        analyze_command = [
            sys.executable, os.path.join("video_processing", "test_serve_analyzer.py"),
            "--video_input", video_path,
            "--json_input", json_input_path,
            "--output_dir", analysis_output_path,
            "--hit_dist", str(args.hit_dist),
            "--log_file", log_file_path,
            "--log_mode", "a"
        ]
        result_analyze = subprocess.run(analyze_command)

        if result_analyze.returncode != 0:
            print(f"  [錯誤] 分析階段失敗！詳情請見日誌檔案: {log_file_path}"); continue
        print("  [完成] 分析成功！")
        print(f"  [日誌] 完整處理紀錄已儲存至: {log_file_path}")

    print(f"\n--- 所有影片處理完畢！總耗時: {datetime.now() - script_start_time} ---")

if __name__ == '__main__':
    main()