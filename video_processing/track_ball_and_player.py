# video_processing/track_ball_and_player.py
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import sys
import json
from datetime import datetime

# --- 專案路徑設定 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="【中文版】對沙灘排球影片進行球、球員追蹤及姿態估計。")
    parser.add_argument("--input", type=str, required=True, help="輸入的短影片片段路徑")
    parser.add_argument("--output_dir", type=str, default="output_data/tracking_output", help="追蹤結果的輸出根目錄")
    parser.add_argument("--ball_model", type=str, default="models/ball_best.pt", help="排球偵測模型路徑")
    parser.add_argument("--player_model", type=str, default="models/yolov8s-pose.pt", help="球員偵測與姿態估計模型路徑")
    parser.add_argument("--conf", type=float, default=0.3, help="物件偵測的置信度閾值")
    parser.add_argument("--device", type=str, default="0", help="推理設備: 'cpu' or GPU id (e.g., '0')")
    parser.add_argument("--config_file_name", type=str, default="court_config.json", help="場地設定檔名稱")
    parser.add_argument("--save_annotated_frames", action="store_true", help="[偵錯功能] 是否將「已標註」的每一幀儲存為圖片，用於視覺檢查。")
    parser.add_argument("--save_original_frames", action="store_true", help="[訓練功能] 是否將「未標註」的原始每一幀儲存為圖片，用於模型訓練。")
    parser.add_argument("--log_file", type=str, default=None, help="將日誌輸出到指定的檔案路徑。")
    parser.add_argument("--log_mode", type=str, default='w', choices=['w', 'a'], help="日誌檔案的寫入模式 ('w' for write, 'a' for append).")
    return parser.parse_args()

# --- 輔助函數 (維持不變) ---
def detect_ball(frame, ball_model, conf_thresh, background_ball_zones):
    detected_balls = []
    try:
        results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        if not results or not results[0].boxes: return detected_balls
        for box in results[0].boxes:
            if box.xyxy is None or len(box.xyxy) == 0: continue
            coords = box.xyxy[0].cpu().numpy()
            if len(coords) < 4: continue
            x1, y1, x2, y2 = map(int, coords)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            is_in_background_zone = False
            if background_ball_zones:
                for zone in background_ball_zones:
                    if zone.get('x1') is not None and zone['x1'] <= center_x <= zone['x2'] and zone['y1'] <= center_y <= zone['y2']:
                        is_in_background_zone = True; break
            detected_balls.append({
                "box_coords": [x1, y1, x2, y2], "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": [center_x, center_y], "is_in_background_zone": is_in_background_zone
            })
    except Exception as e: print(f"!! 在 detect_ball 函數中發生預期外的錯誤: {e}")
    return detected_balls

def detect_and_filter_players(frame, player_pose_model, conf_thresh, court_boundary_np, exclusion_zones_np, court_center_xy):
    all_candidates = []
    try:
        results = player_pose_model(frame, conf=conf_thresh, classes=[0], verbose=False)
        if not results or not results[0].boxes or not results[0].keypoints: return all_candidates
        for i in range(len(results[0].boxes)):
            box, kpts = results[0].boxes[i], results[0].keypoints[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1+x2)/2), float((y1+y2)/2))
            in_exclusion = False
            if exclusion_zones_np:
                for zone_np in exclusion_zones_np:
                    if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0: in_exclusion = True; break
            if in_exclusion: continue
            is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0 if court_boundary_np is not None else False
            dist_to_center = np.linalg.norm(np.array(center_pt) - np.array(court_center_xy)) if court_center_xy else float('inf')
            keypoints_xyc_list = []
            if kpts.xy is not None and kpts.conf is not None:
                kpts_xy, kpts_conf = kpts.xy[0].cpu().numpy(), kpts.conf[0].cpu().numpy()
                for kp_idx in range(kpts_xy.shape[0]): keypoints_xyc_list.append([float(kpts_xy[kp_idx, 0]), float(kpts_xy[kp_idx, 1]), float(kpts_conf[kp_idx])])
            all_candidates.append({
                "box_coords": [x1, y1, x2, y2], "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": list(center_pt), "is_inside_court": bool(is_inside),
                "distance_to_center": float(dist_to_center), "pose_keypoints": keypoints_xyc_list
            })
    except Exception as e: print(f"!! 在 detect_and_filter_players 函數中發生預期外的錯誤: {e}")
    all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    return all_candidates[:4]

def draw_detections(frame, balls, players, court_poly_np, exclusion_zones_np):
    if court_poly_np is not None: cv2.polylines(frame, [court_poly_np], True, (0, 255, 0), 2)
    if exclusion_zones_np:
        for zone_np in exclusion_zones_np: cv2.polylines(frame, [zone_np], True, (255, 0, 255), 2)
    for ball in balls:
        x1, y1, x2, y2 = ball['box_coords']
        color = (255, 192, 203) if ball.get('is_in_background_zone', False) else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for player in players:
        x1, y1, x2, y2 = player['box_coords']
        color = (0, 255, 255) if player['is_inside_court'] else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        kpts = player.get('pose_keypoints')
        if not kpts: continue
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        for bone in skeleton:
            idx1, idx2 = bone[0] - 1, bone[1] - 1
            if len(kpts) > max(idx1, idx2) and kpts[idx1][2] > 0.5 and kpts[idx2][2] > 0.5:
                pt1, pt2 = (int(kpts[idx1][0]), int(kpts[idx1][1])), (int(kpts[idx2][0]), int(kpts[idx2][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        for x, y, conf in kpts:
            if conf > 0.5: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

def main():
    args = parse_args()
    log_file_path = args.log_file
    original_stdout = sys.stdout
    log_file_handler = None
    if log_file_path:
        log_file_handler = open(log_file_path, args.log_mode, encoding='utf-8-sig')
        sys.stdout = log_file_handler
    
    try:
        print(f"--- 階段 1: 追蹤 (track_ball_and_player.py) ---")
        print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print("--- 步驟 1: 檢查路徑與設定 ---")
        video_base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_video_dir = os.path.join(project_root, args.output_dir, video_base_name)
        os.makedirs(output_video_dir, exist_ok=True)
        annotated_video_path = os.path.join(output_video_dir, f"{video_base_name}_annotated.mp4")
        json_output_path = os.path.join(output_video_dir, f"{video_base_name}_all_frames_data_with_pose.json")
        annotated_frames_dir = None
        if args.save_annotated_frames:
            annotated_frames_dir = os.path.join(output_video_dir, "annotated_frames"); os.makedirs(annotated_frames_dir, exist_ok=True)
            print(f"  [資訊] 將儲存「已標註」的圖片至: {annotated_frames_dir}")
        original_frames_dir = None
        if args.save_original_frames:
            original_frames_dir = os.path.join(output_video_dir, "original_frames_for_training"); os.makedirs(original_frames_dir, exist_ok=True)
            print(f"  [資訊] 將儲存「原始」的圖片至: {original_frames_dir}")
        config_path = os.path.join(project_root, args.config_file_name)
        ball_model_path = os.path.join(project_root, args.ball_model)
        player_model_path = os.path.join(project_root, args.player_model)
        for path, name in [(config_path, "場地設定"), (ball_model_path, "球偵測模型"), (player_model_path, "姿態模型")]:
            if not os.path.exists(path): print(f"[致命錯誤] 找不到 {name} 檔案！路徑: {path}"); sys.exit(1)
        print("[成功] 所有設定與模型檔案都存在。")
        print("\n--- 步驟 2: 載入 AI 模型 ---")
        device = f"cuda:{args.device}" if args.device.isdigit() else "cpu"
        ball_model = YOLO(ball_model_path).to(device); player_model = YOLO(player_model_path).to(device)
        print("[成功] 模型載入成功！")
        with open(config_path, 'r', encoding='utf-8') as f_config: court_config = json.load(f_config)
        court_boundary_np = np.array(court_config['court_boundary_polygon'], dtype=np.int32) if court_config.get('court_boundary_polygon') else None
        exclusion_zones_np = [np.array(zone, dtype=np.int32) for zone in court_config.get('exclusion_zones', [])]
        court_center_xy = None
        if court_boundary_np is not None:
            M = cv2.moments(court_boundary_np)
            if M["m00"] != 0: court_center_xy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print("\n--- 步驟 3: 初始化影片處理器 ---")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened(): print(f"[致命錯誤] 無法開啟輸入影片！路徑: {args.input}"); sys.exit(1)
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [資訊] 影片屬性: {w}x{h} @ {fps:.2f} FPS")
        writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened(): print("[警告] 影片寫入器開啟失敗！");
        else: print("[成功] 影片寫入器已開啟。")
        print("\n--- 步驟 4: 開始逐幀分析 ---")
        
        # ✨ 核心修正：在這裡初始化 all_frames_data 和 frame_id_counter ✨
        all_frames_data = []
        frame_id_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: print("\n  [資訊] 影片已讀取完畢。"); break
            if frame_id_counter % 90 == 0: print(f"  [資訊] 正在處理第 {frame_id_counter} 幀...")
            if args.save_original_frames and original_frames_dir:
                cv2.imwrite(os.path.join(original_frames_dir, f"frame_{frame_id_counter:05d}.jpg"), frame)
            balls = detect_ball(frame, ball_model, args.conf, court_config.get("background_ball_zones", []))
            players = detect_and_filter_players(frame, player_model, args.conf, court_boundary_np, exclusion_zones_np, court_center_xy)
            all_frames_data.append({"frame_id": frame_id_counter, "ball_detections": balls, "player_detections": players})
            draw_detections(frame, balls, players, court_boundary_np, exclusion_zones_np)
            if args.save_annotated_frames and annotated_frames_dir:
                cv2.imwrite(os.path.join(annotated_frames_dir, f"frame_{frame_id_counter:05d}.jpg"), frame)
            writer.write(frame); frame_id_counter += 1
        print("\n--- 步驟 5: 完成與儲存 ---")
        cap.release(); writer.release(); cv2.destroyAllWindows()
        with open(json_output_path, 'w', encoding='utf-8') as f_json: json.dump(all_frames_data, f_json, indent=2)
        print("[成功] JSON 數據已儲存。")
        try:
            file_size = os.path.getsize(annotated_video_path)
            print(f"[成功] 影片檔案已儲存。最終大小: {file_size / 1024:.2f} KB")
            if file_size == 0: print("[警告] 輸出影片檔案大小為 0 KB。")
        except OSError as e: print(f"[錯誤] 無法檢查輸出影片的檔案大小: {e}")
    finally:
        if log_file_handler:
            sys.stdout = original_stdout
            log_file_handler.close()

if __name__ == '__main__':
    main()