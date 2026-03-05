import argparse
import json
import os

import cv2
import numpy as np

g_window_name = "Court Config Generator"
g_points = []
g_display_frame = None


def mouse_callback(event, x, y, _flags, _param):
    global g_points
    if event == cv2.EVENT_LBUTTONDOWN:
        g_points.append((x, y))


def draw_points_and_lines(frame, points, color, close_polygon=False):
    for point in points:
        cv2.circle(frame, point, 4, color, -1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], close_polygon, color, 2)


def get_polygon_from_user(base_frame, zone_name, min_points, max_points):
    global g_points, g_display_frame
    g_points = []
    cv2.setMouseCallback(g_window_name, mouse_callback)

    while True:
        g_display_frame = base_frame.copy()
        color = (0, 255, 0) if zone_name == "boundary" else (0, 0, 255)
        draw_points_and_lines(g_display_frame, g_points, color, close_polygon=len(g_points) >= 3)

        cv2.putText(
            g_display_frame,
            f"{zone_name}: click {min_points}-{max_points} points",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            g_display_frame,
            "n=confirm r=reset q=quit",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow(g_window_name, g_display_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            g_points = []
        elif key == ord("q"):
            return None
        elif key == ord("n"):
            if min_points <= len(g_points) <= max_points:
                return [[int(x), int(y)] for x, y in g_points]
            print(f"Need {min_points}-{max_points} points, got {len(g_points)}.")


def get_point_from_user(base_frame, prompt):
    points = get_polygon_from_user(base_frame, "net_point", 1, 1)
    if points is None:
        return None
    point = points[0]
    print(prompt)
    return point


def main(video_path, config_save_path):
    final_config = {
        "court_boundary_polygon": [],
        "exclusion_zones": [],
        "net_y": None,
        "background_ball_zones": [],
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: unable to open video: {video_path}")
        return 1
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: unable to read first frame.")
        return 1

    cv2.namedWindow(g_window_name)

    boundary_points = get_polygon_from_user(first_frame, "boundary", 4, 4)
    if boundary_points is None:
        print("Canceled.")
        cv2.destroyAllWindows()
        return 1
    final_config["court_boundary_polygon"] = boundary_points

    current_frame = first_frame.copy()
    cv2.polylines(current_frame, [np.array(boundary_points, dtype=np.int32)], True, (0, 255, 0), 2)

    while True:
        view = current_frame.copy()
        cv2.putText(view, "Step 2/4 exclusion zones: a=add n=next q=quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(g_window_name, view)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("a"):
            exclusion_points = get_polygon_from_user(current_frame, "exclusion", 3, 20)
            if exclusion_points is None:
                print("Canceled.")
                cv2.destroyAllWindows()
                return 1
            final_config["exclusion_zones"].append(exclusion_points)
            cv2.polylines(current_frame, [np.array(exclusion_points, dtype=np.int32)], True, (0, 0, 255), 2)
        elif key == ord("n"):
            break
        elif key == ord("q"):
            print("Canceled.")
            cv2.destroyAllWindows()
            return 1

    net_point = get_point_from_user(current_frame, "Step 3/4: click any point on net centerline.")
    if net_point is None:
        print("Canceled.")
        cv2.destroyAllWindows()
        return 1
    final_config["net_y"] = int(net_point[1])
    cv2.line(current_frame, (0, final_config["net_y"]), (first_frame.shape[1], final_config["net_y"]), (255, 255, 0), 2)

    while True:
        view = current_frame.copy()
        cv2.putText(view, "Step 4/4 background ball zones: a=add n=finish q=quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(g_window_name, view)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("a"):
            zone_points = get_polygon_from_user(current_frame, "ball_zone", 2, 2)
            if zone_points is None:
                print("Canceled.")
                cv2.destroyAllWindows()
                return 1
            x1, y1 = zone_points[0]
            x2, y2 = zone_points[1]
            zone = {
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
            }
            final_config["background_ball_zones"].append(zone)
            cv2.rectangle(current_frame, (zone["x1"], zone["y1"]), (zone["x2"], zone["y2"]), (255, 0, 255), 2)
        elif key == ord("n"):
            break
        elif key == ord("q"):
            print("Canceled.")
            cv2.destroyAllWindows()
            return 1

    cv2.destroyAllWindows()

    try:
        save_dir = os.path.dirname(config_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4)
    except Exception as exc:
        print(f"Error: failed to save config: {exc}")
        return 1

    print("Config created successfully:")
    print(json.dumps(final_config, indent=2))
    print(f"Saved to: {os.path.abspath(config_save_path)}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Generate court config from a sample video frame.")
    parser.add_argument("--video_path", required=True, help="Path to sample video.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "court_config.json"),
        help="Path to output JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.video_path, args.output))
