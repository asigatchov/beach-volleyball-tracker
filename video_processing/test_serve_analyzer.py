# video_processing/test_serve_analyzer.py
import cv2, json, os, argparse, sys
from datetime import datetime

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
from event_analyzer import find_serve_by_pose_and_toss, is_player_behind_baseline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_input", type=str, required=True)
    parser.add_argument("--json_input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_data/test_output")
    parser.add_argument("--hit_dist", type=float, default=80)
    parser.add_argument("--wrist_dist", type=float, default=50)
    parser.add_argument("--toss_vel", type=float, default=5.0)
    # ✨ 新增日誌參數的定義
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_mode", type=str, default='w', choices=['w', 'a'])
    return parser.parse_args()

def draw_tracked_objects(frame, frame_data):
    for ball in frame_data.get('ball_detections', []):
        x1, y1, x2, y2 = ball['box_coords']
        color = (255, 192, 203) if ball.get('is_in_background_zone', False) else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for player in frame_data.get('player_detections', []):
        x1, y1, x2, y2 = player['box_coords']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

def main():
    args = parse_args()
    log_file_handler = None
    original_stdout = sys.stdout
    if args.log_file:
        try:
            log_file_handler = open(args.log_file, args.log_mode, encoding='utf-8-sig')
            sys.stdout = log_file_handler
        except Exception as e:
            sys.stdout = original_stdout
            print(f"無法開啟日誌檔案: {e}")
            pass
            
    try:
        print(f"\n\n--- 階段 2: 發球分析 (test_serve_analyzer.py) ---\n開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50)
        
        try:
            with open(args.json_input, 'r', encoding='utf-8') as f: all_frames_data = json.load(f)
            print(f"[成功] 已從 {args.json_input} 載入 JSON 數據")
        except FileNotFoundError:
            print(f"[致命錯誤] 找不到 JSON 檔案: {args.json_input}"); sys.exit(1)
            
        try:
            with open(os.path.join(project_root, "court_config.json"), 'r') as f: court_config = json.load(f)
            court_polygon = court_config.get("court_boundary_polygon")
        except FileNotFoundError:
            print("[警告] 找不到 court_config.json，發球區過濾器將被停用。")
            court_polygon = None
            
        config = { "hit_dist_thresh": args.hit_dist, "wrist_dist_thresh": args.wrist_dist, "toss_upward_vel_thresh": args.toss_vel }
        print(f"[資訊] 使用以下參數進行分析: {config}")
        
        serve_events = find_serve_by_pose_and_toss(all_frames_data, config, court_polygon)
        
        if not serve_events: print("[警告] 未偵測到完整的發球序列。")
        else: print(f"[成功] 偵測到發球事件，位於第 {serve_events[0]['frame_id']} 幀！")
            
        video_base_name = os.path.splitext(os.path.basename(args.video_input))[0]
        output_dir = os.path.join(project_root, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{video_base_name}_analysis.mp4")
        
        print(f"[資訊] 正在產生視覺化影片於: {output_video_path}")
        cap = cv2.VideoCapture(args.video_input)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        events_by_frame = {e['frame_id']: e for e in serve_events}
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx < len(all_frames_data): draw_tracked_objects(frame, all_frames_data[frame_idx])
            if frame_idx in events_by_frame:
                player_data = events_by_frame[frame_idx]['server_player_data']
                center = tuple(int(c) for c in player_data['center_point'])
                cv2.circle(frame, center, 80, (0, 0, 255), 3)
                cv2.putText(frame, "SERVE DETECTED!", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
            writer.write(frame); frame_idx += 1
            
        cap.release(); writer.release()
        print("[成功] 視覺化影片已儲存。")
    finally:
        if log_file_handler:
            sys.stdout = original_stdout
            log_file_handler.close()

if __name__ == '__main__':
    main()