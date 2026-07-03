import argparse
import os
import torch
import supervision as sv
from ultralytics import YOLO


class FootballTracker:
    """
    A tracking pipeline that uses YOLO for detection and ByteTrack for tracking,
    annotating players, referees, and the ball on football pitch footage.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        minimum_matching_threshold: float = 0.7,
        track_activation_threshold: float = 0.2,
        minimum_consecutive_frames: int = 10,
        lost_track_buffer: int = 900,
        ball_id: int = 0,
        nms_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the tracker with a YOLO model and ByteTrack configuration.

        Parameters
        ----------
        model_path : str
            Path to the YOLO model file (.pt).
        device : str | None
            Target device for execution (e.g. 'cpu', 'cuda').
        minimum_matching_threshold : float
            ByteTrack matching threshold.
        track_activation_threshold : float
            ByteTrack activation threshold.
        minimum_consecutive_frames : int
            ByteTrack minimum consecutive frames parameter.
        lost_track_buffer : int
            ByteTrack lost track buffer size.
        ball_id : int
            The class ID corresponding to the ball in YOLO detections.
        nms_threshold : float
            Non-max suppression threshold for detections.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLO model from {model_path} onto device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.ball_id = ball_id
        self.nms_threshold = nms_threshold

        # Initialize ByteTrack
        self.tracker = sv.ByteTrack(
            minimum_matching_threshold=minimum_matching_threshold,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            lost_track_buffer=lost_track_buffer,
        )

        # Configured annotators using notebook colors
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"),
            base=25,
            height=21,
            outline_thickness=1,
        )

    def track_video(
        self,
        source_video_path: str,
        target_video_path: str,
        conf: float = 0.4,
    ) -> list[dict]:
        """
        Run the tracking pipeline on the given input video, save the annotated result,
        and return the tracking/detection data.

        Parameters
        ----------
        source_video_path : str
            Path to the input video.
        target_video_path : str
            Path to save the annotated output video.
        conf : float
            Confidence threshold for YOLO predictions.

        Returns
        -------
        list[dict]
            A list of dictionaries, one per frame, containing ball and player detections.
        """
        if not os.path.exists(source_video_path):
            raise FileNotFoundError(
                f"Source video not found at: {source_video_path}"
            )

        print(f"Processing video: {source_video_path}")
        print(f"Target output: {target_video_path}")

        video_info = sv.VideoInfo.from_video_path(source_video_path)

        # Ensure directory for output video exists
        target_dir = os.path.dirname(target_video_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        self.tracker.reset()
        tracking_data = []

        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_video_path)):
                # YOLO Inference
                results = self.model.predict(
                    frame, conf=conf, device=self.device, verbose=False
                )

                detections = sv.Detections.from_ultralytics(results[0])

                # Filter ball detections
                ball_detections = detections[detections.class_id == self.ball_id]
                ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

                # Filter all other detections (players + referees + goalkeeper)
                all_detections = detections[detections.class_id != self.ball_id]
                all_detections = all_detections.with_nms(
                    threshold=self.nms_threshold, class_agnostic=True
                )

                # Shift class ID to align class indices with supervision color palette
                all_detections.class_id -= 1

                # Update tracker
                all_detections = self.tracker.update_with_detections(
                    all_detections
                )

                # Format ball data
                ball_list = []
                for bbox, confidence in zip(ball_detections.xyxy, ball_detections.confidence):
                    ball_list.append({
                        "bbox": bbox.tolist(),
                        "confidence": float(confidence),
                    })

                # Format player/referee/goalkeeper data
                player_list = []
                if all_detections.tracker_id is not None:
                    for bbox, tracker_id, class_id, confidence in zip(
                        all_detections.xyxy,
                        all_detections.tracker_id,
                        all_detections.class_id,
                        all_detections.confidence
                    ):
                        player_list.append({
                            "bbox": bbox.tolist(),
                            "tracker_id": int(tracker_id),
                            "class_id": int(class_id + 1),  # Original YOLO class ID
                            "shifted_class_id": int(class_id),  # Shifted class ID
                            "confidence": float(confidence),
                        })

                tracking_data.append({
                    "frame_index": frame_idx,
                    "ball": ball_list,
                    "players": player_list,
                })

                # Format tracker ID labels
                labels = [
                    f"#{tracker_id}" for tracker_id in all_detections.tracker_id
                ]

                # Annotate the frame
                annotated = frame.copy()
                annotated = self.ellipse_annotator.annotate(
                    annotated, all_detections
                )
                annotated = self.label_annotator.annotate(
                    annotated, all_detections, labels=labels
                )
                annotated = self.triangle_annotator.annotate(
                    annotated, ball_detections
                )

                sink.write_frame(annotated)

        print(f"Video processing finished. Output saved to {target_video_path}")
        return tracking_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track football players, referees, and the ball in video using YOLO and ByteTrack."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the fine-tuned YOLO model (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to save the processed output video file.",
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
        help="Execution device (e.g. cpu, cuda, or device index).",
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

    tracker = FootballTracker(
        model_path=args.model,
        device=args.device,
        minimum_matching_threshold=args.min_matching_thresh,
        track_activation_threshold=args.track_activation_thresh,
        minimum_consecutive_frames=args.min_consecutive_frames,
        lost_track_buffer=args.lost_track_buf,
        ball_id=args.ball_id,
        nms_threshold=args.nms_threshold,
    )

    tracker.track_video(
        source_video_path=args.source,
        target_video_path=args.target,
        conf=args.conf,
    )
