# video_processing/test_serve_detection.py
import json
import os
import sys
import cv2
import numpy as np

# --- Configure Python import paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import functions from modules ---
try:
    from court_definition.court_config_generator import load_court_geometry
    from video_processing.event_analyzer import (
        get_player_center,
        get_baselines_from_ordered_corners,
        is_point_outside_line_segment_extended,
        is_player_behind_baseline,
        find_serve_events
    )
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

# --- (Optional) Visualization helper function ---
def visualize_court_and_points(image_to_draw_on, court_polygon, baselines, points_to_test_dict, title="Court Visualization"):
    """Draw court boundary, baselines, and test points on an image."""
    vis_img = image_to_draw_on.copy()
    # Draw court boundary (blue)
    if court_polygon and len(court_polygon) == 4:
        cv2.polylines(vis_img, [np.array(court_polygon, dtype=np.int32)], True, (255, 0, 0), 2)

    # Draw baselines (red)
    if baselines and baselines[0] and baselines[1]: # far_baseline, near_baseline
        cv2.line(vis_img, baselines[0][0], baselines[0][1], (0, 0, 255), 2) # Far
        cv2.line(vis_img, baselines[1][0], baselines[1][1], (0, 0, 255), 2) # Near

    # Draw test points
    for point_name, data in points_to_test_dict.items():
        coords = data["coords"]
        is_behind = data.get("is_behind_baseline_result", None) # Get test result
        color = (0, 255, 0) if is_behind is True else ((0, 0, 255) if is_behind is False else (255,255,255)) # Green: True, Red: False, White: not tested
        cv2.circle(vis_img, coords, 7, color, -1)
        cv2.putText(vis_img, point_name, (coords[0] + 10, coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow(title, vis_img)
    print(f"Displaying '{title}'. Press any key to continue to the next test or finish...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main test runner ---
def run_all_tests():
    print("--- Starting function tests for event_analyzer.py ---")

    # --- Test parameter setup ---
    video_name_for_data = "123"  # <<--- ★★★ Change to the base filename used to generate _all_frames_data.json ★★★
    config_file_name = "court_config.json"
    
    # Basic video info (must match the video used to generate _all_frames_data.json)
    test_video_fps = 25      # <<--- ★★★ Change to your video's FPS ★★★
    test_video_width = 640    # <<--- ★★★ Change to your video width ★★★
    test_video_height = 640    # <<--- ★★★ Change to your video height ★★★

    # File paths
    all_frames_data_file_path = os.path.join(project_root, "output_data", "tracking_output",
                                           f"{video_name_for_data}",
                                           f"{video_name_for_data}_all_frames_data.json")
    court_config_file_path = os.path.join(project_root, config_file_name)
    sample_video_for_vis_path = os.path.join(project_root, "input_video", f"{video_name_for_data}.avi") # Assume same base name as JSON

    # Load court config
    court_geometry = load_court_geometry(config_load_path=court_config_file_path)
    if court_geometry is None or "court_boundary_polygon" not in court_geometry:
        print(f"Error: Unable to load valid court boundary info from '{court_config_file_path}'. Test aborted.")
        return
    court_polygon = court_geometry["court_boundary_polygon"]
    if not isinstance(court_polygon, list) or len(court_polygon) != 4:
        print(f"Error: court_boundary_polygon format is invalid or does not contain 4 points. Test aborted.")
        return
        
    # Estimate court center (used by is_player_behind_baseline)
    court_center_approx = None
    try:
        poly_np = np.array(court_polygon, dtype=np.int32)
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            court_center_approx = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        elif court_polygon:
            court_center_approx = (
                int(sum(p[0] for p in court_polygon) / len(court_polygon)),
                int(sum(p[1] for p in court_polygon) / len(court_polygon))
            )
    except Exception: pass # Ignore calculation errors; downstream functions handle None


    # --- 1. Test get_player_center ---
    print("\n--- 1. Testing get_player_center ---")
    box1_pc = (100, 100, 200, 200, 0.9)
    center1_x_pc, center1_y_pc = get_player_center(box1_pc)
    print(f"  Box1: {box1_pc} -> Center: ({center1_x_pc}, {center1_y_pc}) (expected: 150.0, 150.0)")
    assert center1_x_pc == 150.0 and center1_y_pc == 150.0, "get_player_center test failed for box1"
    
    box2_pc = [50, 50, 150, 100] # Test case without confidence score
    center2_x_pc, center2_y_pc = get_player_center(box2_pc)
    print(f"  Box2: {box2_pc} -> Center: ({center2_x_pc}, {center2_y_pc}) (expected: 100.0, 75.0)")
    assert center2_x_pc == 100.0 and center2_y_pc == 75.0, "get_player_center test failed for box2"
    print("get_player_center test passed (basic cases).")


    # --- 2. Test get_baselines_from_ordered_corners ---
    print("\n--- 2. Testing get_baselines_from_ordered_corners ---")
    # Use court_polygon loaded from court_config.json
    far_baseline, near_baseline = get_baselines_from_ordered_corners(court_polygon)
    if far_baseline and near_baseline:
        print(f"  Extracted from court boundary {court_polygon}:")
        print(f"    Far baseline (P0-P3): {far_baseline}")
        print(f"    Near baseline (P1-P2): {near_baseline}")
        # You can manually verify whether P0, P1, P2, P3 map correctly for your court_polygon
    else:
        print(f"  Error: Unable to extract baselines from {court_polygon}.")
    # You can draw these two lines in visualization to verify


    # --- 3. & 4. Test is_point_outside_line_segment_extended and is_player_behind_baseline ---
    print("\n--- 3. & 4. Testing is_player_behind_baseline (and helper functions) ---")
    # Load first frame for visualization
    cap_vis = cv2.VideoCapture(sample_video_for_vis_path)
    vis_frame = None
    if cap_vis.isOpened():
        ret_vis, vis_frame = cap_vis.read()
        if not ret_vis: vis_frame = None
    cap_vis.release()

    if vis_frame is None:
        print(f"Warning: Unable to load the first frame from '{sample_video_for_vis_path}' for visualization.")
        # Logic tests can still run without visualization
        vis_frame_for_drawing = np.zeros((test_video_height, test_video_width, 3), dtype=np.uint8) # Create black background
    else:
        vis_frame_for_drawing = vis_frame

    # ★★★★★ Carefully design the following test points for your court_polygon and camera view ★★★★★
    points_to_test_ipbb = {
        "InsideCenter": {"coords": court_center_approx if court_center_approx else (test_video_width//2, test_video_height//2)},
        "BehindNearBaseline": {"coords": ((near_baseline[0][0] + near_baseline[1][0]) // 2, 
                                          (near_baseline[0][1] + near_baseline[1][1]) // 2 + 30) if near_baseline else (0,0)},
        "BehindFarBaseline": {"coords": ((far_baseline[0][0] + far_baseline[1][0]) // 2, 
                                         (far_baseline[0][1] + far_baseline[1][1]) // 2 - 30) if far_baseline else (0,0)},
        "OutsideSidelineNear": {"coords": (near_baseline[0][0] - 50, (near_baseline[0][1] + near_baseline[1][1]) // 2) if near_baseline else (0,0)},
        # Add more boundary test points you consider important...
    }
    
    results_ipbb = {}
    for name, data in points_to_test_ipbb.items():
        if data["coords"] == (0,0) and (not far_baseline or not near_baseline):
            print(f"  Skipping test point '{name}' because baselines are undefined.")
            results_ipbb[name] = None # Mark as not executed
            continue
        
        # is_player_behind_baseline needs frame_h, frame_w, and court_center_approx
        result = is_player_behind_baseline(data["coords"], court_polygon, test_video_height, test_video_width, court_center_approx)
        print(f"  Test point '{name}' at {data['coords']} -> is_behind_baseline: {result}")
        results_ipbb[name] = result
        points_to_test_ipbb[name]["is_behind_baseline_result"] = result # Update dict for visualization

    if vis_frame is not None or court_polygon: # Show when there is any reference object
        visualize_court_and_points(vis_frame_for_drawing, court_polygon, (far_baseline, near_baseline), points_to_test_ipbb, "is_player_behind_baseline Test")


    # --- 5. Test find_serve_events ---
    print("\n--- 5. Testing find_serve_events ---")
    if not os.path.exists(all_frames_data_file_path):
        print(f"Error: Data file '{all_frames_data_file_path}' required for serve-event test does not exist. Skipping this test.")
    else:
        try:
            with open(all_frames_data_file_path, 'r') as f:
                all_frames_data_loaded = json.load(f)
                print(f"  Loaded {len(all_frames_data_loaded)} frames from '{all_frames_data_file_path}'.")

            serve_events = find_serve_events(all_frames_data_loaded, court_geometry, 
                                             test_video_fps, test_video_width, test_video_height)
            
            if serve_events:
                print(f"  --- Detected {len(serve_events)} serve event(s) ---")
                for idx, event in enumerate(serve_events):
                    p_info = event.get('serving_player_info', {})
                    p_box = p_info.get('box_coords', [0,0,0,0])
                    b_start = event.get('ball_start_position', (0,0))
                    print(f"    Event {idx+1}: Frame {event.get('serve_frame_idx')}, PlayerBox {p_box[:2]}, BallStart {b_start}")
            else:
                print("  No serve events detected in this dataset.")

        except Exception as e:
            print(f"  Error while running find_serve_events: {e}")

    print("\n--- All tests finished ---")

if __name__ == "__main__":
    run_all_tests()
