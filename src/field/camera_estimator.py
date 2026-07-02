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
    3. Estimate homography.
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

        self._detections: Optional[
            PitchDetections
        ] = None

        self._homography: Optional[
            HomographyResult
        ] = None

    # ==========================================================
    # Properties
    # ==========================================================

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

    # ==========================================================
    # Public API
    # ==========================================================

    def initialize(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        Initialize camera estimation.
        """

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
        Update camera estimation.
        """

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

        homography = self._homography_estimator.estimate(
            detections,
            self._pitch,
        )

        self._transformer.update(
            homography,
        )

        self._detections = detections
        self._homography = homography

    def reset(self) -> None:
        """
        Reset camera estimation.
        """

        self._tracker.reset()

        self._transformer.reset()

        self._detections = None
        self._homography = None

    # ==========================================================
    # Coordinate Conversion
    # ==========================================================

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