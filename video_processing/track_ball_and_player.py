#!/usr/bin/env python3
# video_processing/track_ball_and_player.py
import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import sys
import json

# --- 專案路徑設定 ---
# 確保可以從 video_processing/ 找到根目錄的 court_definition/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="對沙灘排球影片進行球、球員追蹤及姿態估計。")
    parser.add_argument("--input", type=str, required=True, help="輸入的短影片片段路徑")
    parser.add_argument("--output_dir", type=str, default="output_data/tracking_output", help="追蹤結果的輸出根目錄")
    parser.add_argument("--ball_model", type=str, default="model/ball_best.pt", help="排球偵測模型路徑")
    parser.add_argument("--player_model", type=str, default="model/yolov8s-pose.pt", help="球員偵測與姿態估計模型路徑")
    parser.add_argument("--conf", type=float, default=0.3, help="物件偵測的置信度閾值")
    parser.add_argument("--device", type=str, default="0", help="推理設備: 'cpu' or GPU id (e.g., '0')")
    parser.add_argument("--config_file_name", type=str, default="court_config.json", help="場地設定檔名稱")
    return parser.parse_args()

def detect_ball(frame, ball_model, conf_thresh, background_ball_zones):
    """偵測畫面中的球，並過濾掉在背景過濾區的球。"""
    results = ball_model(frame, conf=conf_thresh, classes=[0], verbose=False)[0]
    detected_balls = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        is_in_background_zone = False
        if background_ball_zones:
            for zone in background_ball_zones:
                if zone['x1'] <= center_x <= zone['x2'] and zone['y1'] <= center_y <= zone['y2']:
                    is_in_background_zone = True
                    break
        if not is_in_background_zone:
            detected_balls.append({
                "box_coords": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": [center_x, center_y]
            })
    return detected_balls

def detect_and_filter_players(frame, player_pose_model, conf_thresh, court_boundary_np, exclusion_zones_np, court_center_xy):
    """
    偵測球員、提取姿態，並根據規則篩選出最重要的4名球員。
    """
    results = player_pose_model(frame, conf=conf_thresh, classes=[0], verbose=False)[0] # class 0 is person
    
    all_candidates = []
    if results.boxes and results.keypoints:
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            kpts = results.keypoints[i]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_pt = (float((x1+x2)/2), float((y1+y2)/2))

            # 1. 排除區域過濾
            in_exclusion = False
            for zone_np in exclusion_zones_np:
                if cv2.pointPolygonTest(zone_np, center_pt, False) >= 0:
                    in_exclusion = True
                    break
            if in_exclusion:
                continue
            
            # 2. 判斷是否在場內
            is_inside = cv2.pointPolygonTest(court_boundary_np, center_pt, False) >= 0 if court_boundary_np is not None else False
            
            # 3. 計算與場地中心距離
            dist_to_center = np.linalg.norm(np.array(center_pt) - np.array(court_center_xy)) if court_center_xy else float('inf')

            # 提取姿態關鍵點
            keypoints_xyc_list = []
            if kpts.xy is not None and kpts.conf is not None:
                kpts_xy = kpts.xy[0].cpu().numpy()
                kpts_conf = kpts.conf[0].cpu().numpy()
                for kp_idx in range(kpts_xy.shape[0]):
                    keypoints_xyc_list.append([float(kpts_xy[kp_idx, 0]), float(kpts_xy[kp_idx, 1]), float(kpts_conf[kp_idx])])

            all_candidates.append({
                "box_coords": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].cpu().numpy()),
                "center_point": list(center_pt),
                "is_inside_court": bool(is_inside),
                "distance_to_center": float(dist_to_center),
                "pose_keypoints": keypoints_xyc_list
            })

    # 排序篩選：場內優先，然後按距離排序
    # is_inside=True (0) 優先於 is_inside=False (1)
    all_candidates.sort(key=lambda p: (not p['is_inside_court'], p['distance_to_center']))
    
    return all_candidates[:4] # 返回分數最高的4名球員

def draw_detections(frame, balls, players, court_poly_np, exclusion_zones_np):
    """在畫面上繪製所有偵測結果，包括球員姿態。"""
    # 繪製場地和排除區
    if court_poly_np is not None:
        cv2.polylines(frame, [court_poly_np], True, (0, 255, 0), 2)
    for zone_np in exclusion_zones_np:
        cv2.polylines(frame, [zone_np], True, (255, 0, 255), 2)
        
    # 繪製球
    for ball in balls:
        x1, y1, x2, y2 = ball['box_coords']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 繪製球員和姿態
    for player in players:
        x1, y1, x2, y2 = player['box_coords']
        color = (0, 255, 255) if player['is_inside_court'] else (0, 165, 255) # 黃色:場內, 橘色:場外
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 繪製姿態
        kpts = player.get('pose_keypoints')
        if not kpts: continue

        # COCO 17點骨架連接
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
                    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
                    [2, 4], [3, 5], [4, 6], [5, 7]]
        
        for bone in skeleton:
            idx1, idx2 = bone[0] - 1, bone[1] - 1 # 轉為 0-based index
            if kpts[idx1][2] > 0.5 and kpts[idx2][2] > 0.5: # 信心度閾值
                pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # 綠色骨架
        
        for x, y, conf in kpts:
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1) # 紅色關節點

def main():
    args = parse_args()
    
    # --- 輸出目錄設定 ---
    video_base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_video_dir = os.path.join(project_root, args.output_dir, video_base_name)
    os.makedirs(output_video_dir, exist_ok=True)
    annotated_video_path = os.path.join(output_video_dir, f"{video_base_name}_annotated.mp4")
    json_output_path = os.path.join(output_video_dir, f"{video_base_name}_all_frames_data_with_pose.json")

    # --- 載入場地設定 ---
    config_path = os.path.join(project_root, args.config_file_name)
    if not os.path.exists(config_path):
        print(f"錯誤：找不到場地設定檔 '{config_path}'。請先執行 court_config_generator.py。")
        return
    with open(config_path, 'r') as f:
        court_config = json.load(f)

    # 準備幾何資訊
    court_boundary_np = np.array(court_config['court_boundary_polygon'], dtype=np.int32)
    exclusion_zones_np = [np.array(zone, dtype=np.int32) for zone in court_config.get('exclusion_zones', [])]
    M = cv2.moments(court_boundary_np)
    court_center_xy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None
    
    # --- 載入模型 ---
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    ball_model = YOLO(os.path.join(project_root, args.ball_model)).to(device)
    player_model = YOLO(os.path.join(project_root, args.player_model)).to(device)

    # --- 影片處理 ---
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    all_frames_data = []
    frame_id_counter = 0
    print(f"開始處理影片: {args.input}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 偵測球
        balls = detect_ball(frame, ball_model, args.conf, court_config.get("background_ball_zones", []))
        
        # 偵測與篩選球員
        players = detect_and_filter_players(frame, player_model, args.conf, court_boundary_np, exclusion_zones_np, court_center_xy)

        # 儲存該幀數據
        all_frames_data.append({
            "frame_id": frame_id_counter,
            "ball_detections": balls,
            "player_detections": players
        })

        # 繪製結果
        draw_detections(frame, balls, players, court_boundary_np, exclusion_zones_np)
        writer.write(frame)
        
        frame_id_counter += 1
        if frame_id_counter % 30 == 0:
            print(f"  已處理 {frame_id_counter} 幀...")

    # --- 收尾工作 ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # 儲存 JSON 數據
    with open(json_output_path, 'w') as f:
        json.dump(all_frames_data, f, indent=2)

    print("\n處理完成！")
    print(f"帶有標註的影片已儲存至：{annotated_video_path}")
    print(f"所有幀的詳細數據已儲存至：{json_output_path}")

if __name__ == '__main__':
    main()