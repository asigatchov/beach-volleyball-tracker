# video_processing/event_analyzer.py
import numpy as np

# COCO Keypoint indices (for clarity)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10

def is_player_behind_baseline(player_center, court_polygon, frame_height):
    """
    一個簡化的判斷，檢查球員是否大致在場地後方。
    這裡假設攝影機視角相對固定，場地遠端Y座標較小，近端較大。
    """
    if not player_center or not court_polygon:
        return False
        
    px, py = player_center
    
    # 找到場地多邊形y座標的最大值，作為近端底線的大致位置
    # 這是一個粗略的估計，但對於初步篩選足夠
    max_y = 0
    for point in court_polygon:
        if point[1] > max_y:
            max_y = point[1]
            
    # 如果球員的y座標大於（更靠近攝影機）這個最大值，我們就認為他在近端底線後
    # 遠端底線的判斷比較複雜，這裡先簡化，主要處理近端發球
    return py > max_y

def find_serve_events(all_frames_data, velocity_threshold=50, proximity_threshold=50):
    """
    基於姿態分析來尋找發球事件。
    核心邏輯：偵測球員手腕的高速揮動，並檢查揮動時手與球是否接觸。

    Args:
        all_frames_data (list): 從 track_ball_and_player.py 產生的數據列表。
        velocity_threshold (int): 手腕關鍵點在單幀內移動多少像素被視為高速揮動。
        proximity_threshold (int): 揮動時，手腕與球中心的距離在多少像素內被視為接觸。

    Returns:
        list: 偵測到的發球事件列表。
    """
    serve_events = []
    
    # 需要至少兩幀數據來計算速度
    if len(all_frames_data) < 2:
        return []

    print("\n開始基於姿態分析發球事件...")
    for i in range(1, len(all_frames_data)):
        prev_frame = all_frames_data[i-1]
        curr_frame = all_frames_data[i]

        # 該幀必須有球被偵測到
        if not curr_frame.get('ball_detections'):
            continue
        # 簡單起見，我們只考慮信心度最高的那個球
        best_ball = max(curr_frame['ball_detections'], key=lambda b: b['confidence'])
        ball_center = np.array(best_ball['center_point'])

        # 遍歷當前幀的每個球員
        for player_idx, curr_player in enumerate(curr_frame.get('player_detections', [])):
            
            # 找到前一幀中同一個球員的數據 (這裡用索引近似，更複雜的需要ID追蹤)
            if player_idx >= len(prev_frame.get('player_detections', [])):
                continue
            prev_player = prev_frame['player_detections'][player_idx]
            
            # --- 姿態分析條件 ---
            curr_kpts = curr_player.get('pose_keypoints')
            prev_kpts = prev_player.get('pose_keypoints')

            if not curr_kpts or not prev_kpts or len(curr_kpts) != 17 or len(prev_kpts) != 17:
                continue
                
            # 檢查左右手腕
            for wrist_idx, shoulder_idx in [(LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)]:
                # 確保關鍵點可信
                if curr_kpts[wrist_idx][2] > 0.5 and prev_kpts[wrist_idx][2] > 0.5 and curr_kpts[shoulder_idx][2] > 0.5:
                    
                    # 1. 計算手腕速度
                    curr_wrist_pos = np.array(curr_kpts[wrist_idx][:2])
                    prev_wrist_pos = np.array(prev_kpts[wrist_idx][:2])
                    velocity = np.linalg.norm(curr_wrist_pos - prev_wrist_pos)
                    
                    # 2. 檢查速度是否超過閾值 (高速揮臂)
                    if velocity > velocity_threshold:
                        
                        # 3. 檢查擊球點高度 (手腕高於肩膀)
                        wrist_y = curr_kpts[wrist_idx][1]
                        shoulder_y = curr_kpts[shoulder_idx][1]
                        if wrist_y < shoulder_y: # 影像座標系中，y越小越高
                            
                            # 4. 檢查手球接觸 (距離夠近)
                            distance_to_ball = np.linalg.norm(curr_wrist_pos - ball_center)
                            if distance_to_ball < proximity_threshold:
                                
                                # 所有條件滿足，記錄為一個發球事件
                                event = {
                                    "frame_id": curr_frame['frame_id'],
                                    "event_type": "SERVE_HIT",
                                    "serving_player_idx": player_idx, # 暫用索引作為ID
                                    "serving_hand": "LEFT" if wrist_idx == LEFT_WRIST else "RIGHT",
                                    "ball_position": list(ball_center),
                                    "wrist_position": list(curr_wrist_pos),
                                    "wrist_velocity": float(velocity),
                                    "hand_ball_distance": float(distance_to_ball)
                                }
                                serve_events.append(event)
                                print(f"  > 偵測到發球事件! 幀: {event['frame_id']}, 球員索引: {event['serving_player_idx']}, 手: {event['serving_hand']}")
                                # 為避免同一幀同一個人的兩隻手都被記錄，可以跳出手的迴圈
                                break 
    
    print(f"分析完成，共偵測到 {len(serve_events)} 個潛在發球擊球點。")
    return serve_events