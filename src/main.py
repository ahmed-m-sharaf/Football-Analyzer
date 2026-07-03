import argparse
import os
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Import local modules
from src.field.pitch_config import PitchConfiguration
from src.field.pitch_detector import RoboflowPitchDetector
from src.field.camera_estimator import CameraEstimator
from src.team_assigner.team_assigner import TeamAssigner
from src.analytics.match_analytics import MatchAnalytics
from src.visualization.match_visualizer import MatchVisualizer


def run_pipeline(
    model_path: str,
    source_video: str,
    target_video: str,
    conf: float = 0.4,
    device: str | None = None,
    roboflow_key: str | None = None,
    min_matching_thresh: float = 0.7,
    track_activation_thresh: float = 0.2,
    min_consecutive_frames: int = 10,
    lost_track_buf: int = 900,
    ball_id: int = 0,
    nms_threshold: float = 0.5,
    progress_callback = None,
) -> None:
    """
    Executes the complete integrated football analytics pipeline on a video.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    model.to(device)

    if not os.path.exists(source_video):
        raise FileNotFoundError(f"Input video not found: {source_video}")

    video_info = sv.VideoInfo.from_video_path(source_video)
    print(f"Input Video: {source_video} | Size: {video_info.width}x{video_info.height} | FPS: {video_info.fps}")

    # Ensure output directory exists
    output_dir = os.path.dirname(target_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize Camera Estimator if Roboflow API key is available
    camera_estimator = None
    roboflow_api_key = roboflow_key or os.environ.get("ROBOFLOW_API_KEY")
    if roboflow_api_key:
        try:
            print("Initializing Roboflow Pitch Detector...")
            pitch = PitchConfiguration()
            detector = RoboflowPitchDetector(
                api_key=roboflow_api_key,
                model_id="football-field-detection-f07vi/14"
            )
            camera_estimator = CameraEstimator(detector=detector, pitch=pitch)
            print("Camera estimator successfully initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize camera estimator ({e}).")
            print("Running in image-pixel space mode.")
    else:
        print("No Roboflow API Key provided. Running in image-pixel space mode.")

    # Initialize tracker
    tracker = sv.ByteTrack(
        minimum_matching_threshold=min_matching_thresh,
        track_activation_threshold=track_activation_thresh,
        minimum_consecutive_frames=min_consecutive_frames,
        lost_track_buffer=lost_track_buf,
    )

    # Initialize team assigner, analytics, and visualizer
    team_assigner = TeamAssigner()
    analytics = MatchAnalytics(fps=video_info.fps, possession_threshold_meters=2.0)
    visualizer = MatchVisualizer()

    print(f"Processing video: {source_video}...")
    tracker.reset()

    with sv.VideoSink(target_video, video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_video)):
            # 1. Run YOLO inference
            results = model.predict(frame, conf=conf, device=device, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])

            # Filter ball detections
            ball_detections = detections[detections.class_id == ball_id]
            ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

            # Filter player detections (players + referees + goalkeeper)
            all_detections = detections[detections.class_id != ball_id]
            all_detections = all_detections.with_nms(threshold=nms_threshold, class_agnostic=True)

            # Shift class ID to align class indices with supervision tracking index
            all_detections.class_id -= 1

            # Update tracker
            all_detections = tracker.update_with_detections(all_detections)

            # Format player data
            frame_players = []
            if all_detections.tracker_id is not None:
                for bbox, tracker_id, class_id, confidence in zip(
                    all_detections.xyxy,
                    all_detections.tracker_id,
                    all_detections.class_id,
                    all_detections.confidence
                ):
                    frame_players.append({
                        "bbox": bbox.tolist(),
                        "tracker_id": int(tracker_id),
                        "class_id": int(class_id + 1),  # Original YOLO class ID
                        "shifted_class_id": int(class_id),
                        "confidence": float(confidence),
                    })

            # Format ball data
            frame_ball = []
            for bbox, confidence in zip(ball_detections.xyxy, ball_detections.confidence):
                frame_ball.append({
                    "bbox": bbox.tolist(),
                    "confidence": float(confidence),
                })

            # 2. Camera Homography update
            if camera_estimator is not None:
                try:
                    camera_estimator.update(frame)
                except Exception:
                    pass
                transformer = camera_estimator.transformer
            else:
                transformer = None

            # 3. Fit Team Color Assignment on first available frame
            if team_assigner.kmeans is None:
                first_frame_players = {
                    p["tracker_id"]: p for p in frame_players if p["class_id"] == 2
                }
                if len(first_frame_players) >= 2:
                    team_assigner.assign_team_color(frame, first_frame_players)

            # Assign team ID to players and goalkeepers
            player_teams = {}
            for p in frame_players:
                if p["class_id"] in [1, 2]:  # Goalkeeper or standard player
                    team_id = team_assigner.get_player_team(frame, p["bbox"], p["tracker_id"])
                    p["team_id"] = team_id
                    player_teams[p["tracker_id"]] = team_id

            # 4. Calculate Analytics
            metrics = analytics.update(
                players=frame_players,
                ball=frame_ball,
                transformer=transformer,
                frame_idx=frame_idx,
            )

            # 5. Visualization Overlay
            annotated = frame.copy()
            
            # Team colored circles under feet
            annotated = visualizer.draw_player_circles(
                annotated,
                players=frame_players,
                player_teams=player_teams,
                team_colors_kmeans=team_assigner.team_colors,
            )

            # Speed and Distance Covered Tags
            annotated = visualizer.draw_player_tags(
                annotated,
                players=frame_players,
                player_teams=player_teams,
                player_speeds=metrics["player_speeds"],
                player_distances=metrics["player_distances"],
            )

            # Ball Indicator
            annotated = visualizer.draw_ball_marker(annotated, ball=frame_ball)

            # Possession Top HUD
            annotated = visualizer.draw_possession_hud(
                annotated,
                possession_percentages=metrics["possession_percentages"],
            )

            sink.write_frame(annotated)

            if progress_callback is not None:
                progress_callback(frame_idx + 1, video_info.total_frames)

            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}...")

    print(f"Finished processing video. Output saved to: {target_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete football analytics pipeline with player speed, distance, and team possession."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/best.pt",
        help="Path to YOLO model (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/input/08fd33_0.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/videos/output/annotated_08fd33_0.mp4",
        help="Path to save processed output video file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO predictions (default: 0.4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--roboflow-key",
        type=str,
        default=None,
        help="Roboflow API key for field calibration.",
    )
    parser.add_argument(
        "--min-matching-thresh",
        type=float,
        default=0.7,
        help="Minimum matching threshold for ByteTrack (default: 0.7).",
    )
    parser.add_argument(
        "--track-activation-thresh",
        type=float,
        default=0.2,
        help="Track activation threshold for ByteTrack (default: 0.2).",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=10,
        help="Minimum consecutive frames for ByteTrack (default: 10).",
    )
    parser.add_argument(
        "--lost-track-buf",
        type=int,
        default=900,
        help="Lost track buffer for ByteTrack (default: 900).",
    )
    parser.add_argument(
        "--ball-id",
        type=int,
        default=0,
        help="Class ID of the ball in YOLO model (default: 0).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="Non-max suppression threshold (default: 0.5).",
    )

    args = parser.parse_args()

    run_pipeline(
        model_path=args.model,
        source_video=args.source,
        target_video=args.target,
        conf=args.conf,
        device=args.device,
        roboflow_key=args.roboflow_key,
        min_matching_thresh=args.min_matching_thresh,
        track_activation_thresh=args.track_activation_thresh,
        min_consecutive_frames=args.min_consecutive_frames,
        lost_track_buf=args.lost_track_buf,
        ball_id=args.ball_id,
        nms_threshold=args.nms_threshold,
    )
