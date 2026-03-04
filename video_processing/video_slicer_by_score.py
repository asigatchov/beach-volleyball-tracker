# video_slicer_by_score_comparison.py (or your new filename)
import cv2
import os
import argparse
import numpy as np

# --- Config ---
SCORE_ROI_TEAM1 = (280, 29, 59, 51)  # Example value
SCORE_ROI_TEAM2 = (287, 92, 59, 50)  # Example value

def parse_arguments():
    parser = argparse.ArgumentParser(description="Split videos based on image changes in two independent score ROIs.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input long-form video file")
    parser.add_argument("--output_dir", type=str, default="../output_data/video_segments_output_comp_split_v2", help="Root directory to store split video segments")
    parser.add_argument("--min_segment_duration", type=int, default=10, help="Minimum duration of a valid segment (seconds)")
    parser.add_argument("--long_segment_threshold", type=int, default=90, help="Threshold for long segments (seconds)")
    parser.add_argument("--roi_check_interval", type=float, default=0.5, help="How often to check ROI changes (seconds)")
    # Removed --no_change_timeout because the current logic splits based on detected changes
    parser.add_argument("--diff_threshold", type=int, default=600, 
                        help="Image difference threshold (SAD) for a single ROI. Needs tuning.")
    return parser.parse_args()

def finalize_segment_processing(temp_filename, segment_frames_written, fps, min_duration_sec, long_threshold_sec,
                                normal_dir, long_dir, segment_id_counter):
    # (This function is identical to the previous version)
    if not os.path.exists(temp_filename) or segment_frames_written == 0:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError as e: print(f"Error deleting empty temp file {temp_filename}: {e}")
        return

    duration_sec = segment_frames_written / fps
    final_base_name = f"segment_{segment_id_counter:03d}.mp4"

    if duration_sec < min_duration_sec:
        print(f"Segment {final_base_name} ({duration_sec:.1f}s) is too short (<{min_duration_sec}s), deleted.")
        try: os.remove(temp_filename)
        except OSError as e: print(f"Error deleting short segment {temp_filename}: {e}")
    elif duration_sec > long_threshold_sec:
        target_filename = os.path.join(long_dir, final_base_name)
        print(f"Segment {final_base_name} ({duration_sec:.1f}s) is a long segment (>{long_threshold_sec}s), moved to: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"Error moving long segment {temp_filename} -> {target_filename}: {e}")
    else: # Normal segment
        target_filename = os.path.join(normal_dir, final_base_name)
        print(f"Segment {final_base_name} ({duration_sec:.1f}s) is a normal segment, moved to: {target_filename}")
        try: os.rename(temp_filename, target_filename)
        except OSError as e: print(f"Error moving normal segment {temp_filename} -> {target_filename}: {e}")

def get_roi_image(frame, roi_coords, frame_width, frame_height):
    # (This function is the same as previous version)
    x, y, w, h = roi_coords
    if not (0 <= x < frame_width and 0 <= y < frame_height and x + w <= frame_width and y + h <= frame_height and w > 0 and h > 0):
        return None
    roi_img = frame[y:y+h, x:x+w]
    return cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

def main():
    args = parse_arguments()
    # ... (Output directory creation logic is unchanged) ...
    output_root_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_root_abs, exist_ok=True)
    normal_segments_dir = os.path.join(output_root_abs, "normal_segments")
    long_segments_dir = os.path.join(output_root_abs, "long_segments")
    os.makedirs(normal_segments_dir, exist_ok=True)
    os.makedirs(long_segments_dir, exist_ok=True)
    temp_dir = os.path.join(output_root_abs, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    # ... (Video open and FPS retrieval logic is unchanged) ...
    if not cap.isOpened(): print(f"Error: Unable to open video {args.input}"); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: print("Error: Unable to get video FPS."); cap.release(); return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video: {args.input}, FPS: {fps:.2f}")
    print(f"Output to: {output_root_abs}")
    print(f"ROI check interval: {args.roi_check_interval}s, difference threshold (SAD): {args.diff_threshold}")
    print(f"Minimum segment duration: {args.min_segment_duration}s, long segment threshold: {args.long_segment_threshold}s")


    previous_roi1_gray = None
    previous_roi2_gray = None
    
    # is_game_active still marks whether a segment is currently being recorded
    is_game_active = False 
    
    video_writer = None
    current_temp_video_path = None
    segment_id_counter = 0
    frames_written_this_segment = 0

    frame_idx = 0
    roi_check_interval_frames = int(fps * args.roi_check_interval)
    if roi_check_interval_frames == 0: roi_check_interval_frames = 1
    
    # Removed no_change_timeout_checks and frames_since_last_change_or_valid_roi
    # because current logic is "split when changed"; there is no timeout-based ending

    # ROI coordinates (make sure they are within frame bounds)
    rois = {"team1": SCORE_ROI_TEAM1, "team2": SCORE_ROI_TEAM2}
    for team_name, (rx, ry, rw, rh) in rois.items():
        if not (0 <= rx < frame_width and 0 <= ry < frame_height and \
                rx + rw <= frame_width and ry + rh <= frame_height and rw > 0 and rh > 0):
            print(f"Error: {team_name} ROI {rois[team_name]} is out of frame bounds or has invalid size.")
            cap.release()
            return

    first_valid_rois_captured = False # Whether the first valid ROI set has been captured as baseline

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # If a segment is being recorded, write frames
        # This write condition is moved after ROI checks so the trigger frame set is included
        # if is_game_active and video_writer is not None:
        #     video_writer.write(frame)
        #     frames_written_this_segment += 1

        if frame_idx % roi_check_interval_frames == 0:
            current_roi1_gray = get_roi_image(frame, SCORE_ROI_TEAM1, frame_width, frame_height)
            current_roi2_gray = get_roi_image(frame, SCORE_ROI_TEAM2, frame_width, frame_height)

            if current_roi1_gray is None or current_roi2_gray is None:
                print(f"Frame {frame_idx}: One or more ROIs failed to extract, skipping this checkpoint.")
                # If recording, you could end the segment on ROI failure or ignore temporarily
                # Current logic: if recording, continue until next successful ROI comparison
                if is_game_active and video_writer is not None: # Keep continuity even when ROI fails
                    video_writer.write(frame)
                    frames_written_this_segment += 1
                continue # Skip change detection for this checkpoint

            if not first_valid_rois_captured: # Capture first valid ROI set as comparison baseline
                previous_roi1_gray = current_roi1_gray.copy()
                previous_roi2_gray = current_roi2_gray.copy()
                first_valid_rois_captured = True
                print(f"Frame {frame_idx}: Initial ROI state captured.")
                if is_game_active and video_writer is not None: # Keep writing for continuity
                    video_writer.write(frame)
                    frames_written_this_segment += 1
                continue


            roi1_has_changed = False
            diff1 = cv2.absdiff(current_roi1_gray, previous_roi1_gray)
            sad1 = np.sum(diff1)
            if sad1 > args.diff_threshold:
                roi1_has_changed = True

            roi2_has_changed = False
            diff2 = cv2.absdiff(current_roi2_gray, previous_roi2_gray)
            sad2 = np.sum(diff2)
            if sad2 > args.diff_threshold:
                roi2_has_changed = True
            
            # print(f"Frame {frame_idx}: SAD1={sad1}, SAD2={sad2}")

            overall_roi_has_changed = roi1_has_changed or roi2_has_changed
            
            if overall_roi_has_changed:
                print(f"Frame {frame_idx}: ROI change detected (ROI1 changed: {roi1_has_changed}, ROI2 changed: {roi2_has_changed})")
                
                # If a segment was being recorded, finish and save it first
                if is_game_active and video_writer is not None:
                    print(f"--- Ending segment due to ROI change (ID: {segment_id_counter:03d}) ---")
                    video_writer.release()
                    finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                                args.min_segment_duration, args.long_segment_threshold,
                                                normal_segments_dir, long_segments_dir, segment_id_counter)
                    # is_game_active = False # A new one starts immediately, so no need
                
                # Start a new segment
                is_game_active = True # Ensure active state
                segment_id_counter += 1
                current_temp_video_path = os.path.join(temp_dir, f"segment_{segment_id_counter:03d}_temp.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(current_temp_video_path, fourcc, fps, (frame_width, frame_height))
                frames_written_this_segment = 0 # Reset counter for new segment
                print(f"=== Starting new segment due to ROI change (ID: {segment_id_counter:03d}), writing to: {current_temp_video_path} ===")
            
            # Update previous_roi_gray for next comparison
            previous_roi1_gray = current_roi1_gray.copy()
            previous_roi2_gray = current_roi2_gray.copy()
        
        # Frame writing logic moved here to run after ROI checks and segment start/end handling
        # This includes the frame group that triggered change (entire interval from ROI checkpoint)
        if is_game_active and video_writer is not None:
            video_writer.write(frame)
            frames_written_this_segment += 1
            
    # --- After loop ends ---
    if video_writer is not None: # Process final unfinished segment at video end
        print(f"--- Video ended, finishing last segment (ID: {segment_id_counter:03d}) ---")
        video_writer.release()
        finalize_segment_processing(current_temp_video_path, frames_written_this_segment, fps,
                                    args.min_segment_duration, args.long_segment_threshold,
                                    normal_segments_dir, long_segments_dir, segment_id_counter)

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo splitting completed!") # ... (other prints)

if __name__ == "__main__":
    main()
