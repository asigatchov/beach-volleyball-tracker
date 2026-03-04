# court_definition/court_config_generator.py

# ... (Other unchanged code is omitted here, e.g. mouse_callback, get_polygon_from_user, etc.) ...

def main(video_path, config_save_path):
    global g_final_config

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Unable to read the first frame of the video.")
        return

    cv2.namedWindow(g_window_name)

    # 1. Define court boundary
    boundary_points = get_polygon_from_user(first_frame, 'boundary', 4, 4)
    if boundary_points is None:
        print("User canceled operation. Definition aborted.")
        cv2.destroyAllWindows()
        return
    g_final_config["court_boundary_polygon"] = boundary_points

    # Draw defined boundary as base image for subsequent steps
    base_frame_with_boundary = first_frame.copy()
    cv2.polylines(base_frame_with_boundary, [np.array(boundary_points)], True, (0, 255, 0), 2)

    # --- New flow: define exclusion zones (multiple allowed) ---
    current_drawing_frame = base_frame_with_boundary.copy()
    while True:
        prompt = "Step 2/4: Define exclusion zones | Press 'a' add, 'n' next, 'q' quit"
        cv2.putText(current_drawing_frame, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(g_window_name, current_drawing_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'): # Add
            exclusion_points = get_polygon_from_user(current_drawing_frame, 'exclusion', 3, 20)
            if exclusion_points:
                 g_final_config["exclusion_zones"].append(exclusion_points)
                 # Draw newly defined zone on the frame
                 cv2.polylines(current_drawing_frame, [np.array(exclusion_points)], True, (0, 0, 255), 2)
            else: # If 'q' was pressed in get_polygon_from_user
                 print("User canceled operation. Definition aborted.")
                 cv2.destroyAllWindows()
                 return
        elif key == ord('n'): # Next
            break
        elif key == ord('q'):
            print("User canceled operation. Definition aborted.")
            cv2.destroyAllWindows()
            return

    # 3. Define net Y coordinate
    net_point = get_point_from_user(current_drawing_frame, "Step 3/4: Click any point on the net centerline")
    if net_point is None:
        print("User canceled operation. Definition aborted.")
        cv2.destroyAllWindows()
        return
    g_final_config["net_y"] = net_point[1]
    # Draw net line for reference
    cv2.line(current_drawing_frame, (0, net_point[1]), (first_frame.shape[1], net_point[1]), (255, 255, 0), 2)

    # --- New flow: define background-ball filter zones (multiple allowed) ---
    while True:
        prompt = "Step 4/4: Define background ball zones | Press 'a' add, 'n' finish, 'q' quit"
        cv2.putText(current_drawing_frame, "Step 4/4: Define background ball zones", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(current_drawing_frame, "Press 'a' add, 'n' finish, 'q' quit", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
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
            else: # If 'q' was pressed in get_polygon_from_user
                 print("User canceled operation. Definition aborted.")
                 cv2.destroyAllWindows()
                 return
        elif key == ord('n'): # Next / Finish
            break
        elif key == ord('q'):
            print("User canceled operation. Definition aborted.")
            cv2.destroyAllWindows()
            return
            
    cv2.destroyAllWindows()
    print("\n--- Definition completed. Final config below ---")
    print(json.dumps(g_final_config, indent=2))

    # Save config file
    try:
        with open(config_save_path, 'w') as f:
            json.dump(g_final_config, f, indent=4)
        print(f"\nConfig saved successfully to: {os.path.abspath(config_save_path)}")
    except Exception as e:
        print(f"\nError: Failed to save config file: {e}")

# ... (File ending omitted here, including if __name__ == '__main__': block) ...
