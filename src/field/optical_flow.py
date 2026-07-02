from typing import Optional

import cv2
import numpy as np

from .pitch_detector import (
    DetectedKeyPoint,
    PitchDetections,
)


class OpticalFlowTracker:
    """
    Tracks football pitch keypoints using Sparse Lucas-Kanade Optical Flow.

    Features
    --------
    - Lucas-Kanade Optical Flow
    - Forward-Backward consistency check
    - Automatic redetection trigger
    - Keeps keypoint class IDs
    """

    def __init__(
        self,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        criteria: tuple = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        fb_threshold: float = 1.5,
        min_keypoints: int = 6,
        max_frames_without_detection: int = 30,
    ) -> None:

        self.win_size = win_size
        self.max_level = max_level
        self.criteria = criteria

        self.fb_threshold = fb_threshold
        self.min_keypoints = min_keypoints
        self.max_frames_without_detection = (
            max_frames_without_detection
        )

        self.reset()

    @property
    def is_initialized(self) -> bool:
        return (
            self._previous_gray is not None
            and self._previous_detections is not None
        )

    @property
    def needs_redetection(self) -> bool:

        if not self.is_initialized:
            return True

        if (
            self._previous_detections.num_keypoints
            < self.min_keypoints
        ):
            return True

        if (
            self._frames_since_detection
            >= self.max_frames_without_detection
        ):
            return True

        return False


    def initialize(
        self,
        frame: np.ndarray,
        detections: PitchDetections,
    ) -> None:
        """
        Initialize tracker from detector output.
        """

        self._previous_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        self._previous_detections = detections

        self._frames_since_detection = 0

    def reset(self) -> None:

        self._previous_gray: Optional[np.ndarray] = None

        self._previous_detections: Optional[
            PitchDetections
        ] = None

        self._frames_since_detection = 0


    def track(
        self,
        frame: np.ndarray,
    ) -> PitchDetections:

        if not self.is_initialized:
            raise RuntimeError(
                "Tracker has not been initialized."
            )

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        previous_points = (
            self._previous_detections
            .image_points()
            .reshape(-1, 1, 2)
        )


        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )

        if next_points is None:
            self.reset()
            return PitchDetections()

        backward_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            next_points,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )

        tracked_keypoints = []

        for (
            kp,
            prev_point,
            next_point,
            back_point,
            ok_forward,
            ok_backward,
        ) in zip(
            self._previous_detections.keypoints,
            previous_points,
            next_points,
            backward_points,
            status_forward,
            status_backward,
        ):

            if not ok_forward[0]:
                continue

            if not ok_backward[0]:
                continue


            fb_error = np.linalg.norm(
                prev_point[0] - back_point[0]
            )

            if fb_error > self.fb_threshold:
                continue

            x, y = next_point[0]

            tracked_keypoints.append(
                DetectedKeyPoint(
                    class_id=kp.class_id,
                    x=float(x),
                    y=float(y),
                    confidence=kp.confidence,
                )
            )

        detections = PitchDetections(
            tracked_keypoints
        )

        self._previous_gray = current_gray
        self._previous_detections = detections

        self._frames_since_detection += 1

        return detections