# court_definition/court_config_generator.py

# ... (此處省略了其他未變動的程式碼，如 mouse_callback, get_polygon_from_user 等) ...

def main(video_path, config_save_path):
    global g_final_config

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 '{video_path}'")
        return
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("錯誤：無法讀取影片的第一幀。")
        return

    cv2.namedWindow(g_window_name)

    # 1. 定義場地邊界
    boundary_points = get_polygon_from_user(first_frame, 'boundary', 4, 4)
    if boundary_points is None:
        print("使用者取消操作，定義終止。")
        cv2.destroyAllWindows()
        return
    g_final_config["court_boundary_polygon"] = boundary_points

    # 繪製已定義的邊界，作為後續步驟的底圖
    base_frame_with_boundary = first_frame.copy()
    cv2.polylines(base_frame_with_boundary, [np.array(boundary_points)], True, (0, 255, 0), 2)

    # --- ✨ 新流程：定義排除區域 (可定義多個) ---
    current_drawing_frame = base_frame_with_boundary.copy()
    while True:
        prompt = "步驟2/4: 定義排除區 | 按 'a' 新增, 'n' 到下一步, 'q' 退出"
        cv2.putText(current_drawing_frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(g_window_name, current_drawing_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'): # Add
            exclusion_points = get_polygon_from_user(current_drawing_frame, 'exclusion', 3, 20)
            if exclusion_points:
                 g_final_config["exclusion_zones"].append(exclusion_points)
                 # 在圖上繪製新定義的區域
                 cv2.polylines(current_drawing_frame, [np.array(exclusion_points)], True, (0, 0, 255), 2)
            else: # 如果在 get_polygon_from_user 中按了 q
                 print("使用者取消操作，定義終止。")
                 cv2.destroyAllWindows()
                 return
        elif key == ord('n'): # Next
            break
        elif key == ord('q'):
            print("使用者取消操作，定義終止。")
            cv2.destroyAllWindows()
            return

    # 3. 定義網子Y座標
    net_point = get_point_from_user(current_drawing_frame, "步驟3/4: 請點擊網子中心線上的任一點")
    if net_point is None:
        print("使用者取消操作，定義終止。")
        cv2.destroyAllWindows()
        return
    g_final_config["net_y"] = net_point[1]
    # 畫出網子線以供參考
    cv2.line(current_drawing_frame, (0, net_point[1]), (first_frame.shape[1], net_point[1]), (255, 255, 0), 2)

    # --- ✨ 新流程：定義背景球過濾區 (可定義多個) ---
    while True:
        prompt = "步驟4/4: 定義背景球區 | 按 'a' 新增, 'n' 完成, 'q' 退出"
        cv2.putText(current_drawing_frame, "步驟4/4: 定義背景球區", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(current_drawing_frame, "按 'a' 新增, 'n' 完成, 'q' 退出", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(g_window_name, current_drawing_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'): # Add
            zone_points = get_polygon_from_user(current_drawing_frame, "ball_zone", 2, 2)
            if zone_points:
                x1, y1 = zone_points[0]
                x2, y2 = zone_points[1]
                g_final_config["background_ball_zones"].append({
                    "x1": min(x1, x2), "y1": min(y1, y2),
                    "x2": max(x1, x2), "y2": max(y1, y2)
                })
                cv2.rectangle(current_drawing_frame, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (255,0,255), 2)
            else: # 如果在 get_polygon_from_user 中按了 q
                 print("使用者取消操作，定義終止。")
                 cv2.destroyAllWindows()
                 return
        elif key == ord('n'): # Next / Finish
            break
        elif key == ord('q'):
            print("使用者取消操作，定義終止。")
            cv2.destroyAllWindows()
            return
            
    cv2.destroyAllWindows()
    print("\n--- ✅ 定義完成，最終設定如下 ---")
    print(json.dumps(g_final_config, indent=2))

    # 儲存設定檔
    try:
        with open(config_save_path, 'w') as f:
            json.dump(g_final_config, f, indent=4)
        print(f"\n設定已成功儲存至：{os.path.abspath(config_save_path)}")
    except Exception as e:
        print(f"\n錯誤：儲存設定檔失敗：{e}")

# ... (此處省略了檔案結尾的 if __name__ == '__main__': block) ...