from typing import Optional

import numpy as np

from .pitch_config import PitchConfiguration
from .pitch_detector import (
    PitchDetections,
    RoboflowPitchDetector,
)
from .optical_flow import OpticalFlowTracker
from .homography import (
    HomographyEstimator,
    HomographyResult,
)
from .transformer import CoordinateTransformer


class CameraEstimator:
    """
    Estimates the camera geometry throughout a video.

    Responsibilities
    ----------------
    1. Detect pitch keypoints.
    2. Track them using optical flow.
    3. Estimate homography with quality validation and EMA stabilization.
    4. Update coordinate transformer.
    """

    def __init__(
        self,
        detector: RoboflowPitchDetector,
        pitch: PitchConfiguration,
        tracker: Optional[OpticalFlowTracker] = None,
        homography_estimator: Optional[
            HomographyEstimator
        ] = None,
        debug: bool = False,
    ) -> None:

        self._pitch = pitch
        self._detector = detector
        self._tracker = (
            tracker
            if tracker is not None
            else OpticalFlowTracker()
        )
        self._homography_estimator = (
            homography_estimator
            if homography_estimator is not None
            else HomographyEstimator()
        )
        self._transformer = CoordinateTransformer()
        self._detections: Optional[PitchDetections] = None
        self._homography: Optional[HomographyResult] = None
        self._debug = debug
        self._frame_idx = 0

    @property
    def is_initialized(self) -> bool:
        return self._homography is not None

    @property
    def detections(self) -> PitchDetections:
        if self._detections is None:
            raise RuntimeError(
                "Camera has not been initialized."
            )
        return self._detections

    @property
    def homography(self) -> HomographyResult:
        if self._homography is None:
            raise RuntimeError(
                "Camera has not been initialized."
            )
        return self._homography

    @property
    def transformer(self) -> CoordinateTransformer:
        return self._transformer

    def initialize(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Initialize camera estimation.
        """
        self._frame_idx = 0
        detections = self._detector.detect(frame)
        homography = self._homography_estimator.estimate(
            detections,
            self._pitch,
        )
        self._tracker.initialize(
            frame,
            detections,
        )
        self._transformer.update(
            homography,
        )
        self._detections = detections
        self._homography = homography

    def update(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Update camera estimation with quality checks and EMA stabilization.
        """
        self._frame_idx += 1

        if not self.is_initialized:
            self.initialize(frame)
            return

        if self._tracker.needs_redetection:
            detections = self._detector.detect(
                frame,
            )
            self._tracker.initialize(
                frame,
                detections,
            )
        else:
            detections = self._tracker.track(
                frame,
            )

        try:
            candidate = self._homography_estimator.estimate(
                detections,
                self._pitch,
            )
        except Exception as e:
            if self._debug:
                print(f"[CameraEstimator Debug] Frame {self._frame_idx}: Homography estimation failed: {e}")
            # Keep previous valid homography
            return

        # Quality validation check against previous matrix
        prev_matrix = self._homography.matrix
        new_matrix = candidate.matrix

        # Calculate relative Frobenius norm difference
        norm_diff = float(np.linalg.norm(new_matrix - prev_matrix) / np.linalg.norm(prev_matrix))

        is_valid = True
        reason = "Passed"

        if candidate.inlier_ratio < 0.4:
            is_valid = False
            reason = f"Inlier ratio too low ({candidate.inlier_ratio:.3f} < 0.4)"
        elif candidate.reprojection_error > 12.0:
            is_valid = False
            reason = f"Reprojection error too high ({candidate.reprojection_error:.2f} px > 12.0 px)"
        elif norm_diff > 0.25:
            is_valid = False
            reason = f"Relative matrix norm difference too high ({norm_diff:.4f} > 0.25)"

        if self._debug:
            print(
                f"[CameraEstimator Debug] Frame {self._frame_idx} | "
                f"Inliers: {candidate.num_inliers} | "
                f"Inlier Ratio: {candidate.inlier_ratio:.3f} | "
                f"Reproj Error: {candidate.reprojection_error:.2f} px | "
                f"Matrix Diff: {norm_diff:.4f} | "
                f"Decision: {'Accepted (EMA Stabilized)' if is_valid else 'Rejected (' + reason + ')'}"
            )

        if is_valid:
            # Apply Temporal Smoothing (EMA Blend) to stabilize tracking
            alpha = 0.15
            smoothed_matrix = alpha * new_matrix + (1.0 - alpha) * prev_matrix
            homography = HomographyResult(
                matrix=smoothed_matrix,
                mask=candidate.mask,
                reprojection_error=candidate.reprojection_error,
                inlier_ratio=candidate.inlier_ratio,
            )
            self._homography = homography
            self._detections = detections
            self._transformer.update(homography)
        else:
            # Fall back to previous valid homography. Do not update _transformer or _homography.
            pass

    def reset(self) -> None:
        """
        Reset camera estimation.
        """
        self._tracker.reset()
        self._transformer.reset()
        self._detections = None
        self._homography = None
        self._frame_idx = 0

    def image_to_world(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        return self._transformer.image_to_world(
            point,
        )

    def image_to_world_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        return self._transformer.image_to_world_points(
            points,
        )

    def world_to_image(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        return self._transformer.world_to_image(
            point,
        )

    def world_to_image_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        return self._transformer.world_to_image_points(
            points,
        )