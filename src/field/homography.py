from dataclasses import dataclass

import cv2
import numpy as np

from .pitch_config import PitchConfiguration
from .pitch_detector import PitchDetections


@dataclass(frozen=True, slots=True)
class HomographyResult:
    """
    Stores the estimated homography matrix.
    """

    matrix: np.ndarray
    mask: np.ndarray

    @property
    def num_inliers(self) -> int:
        return int(self.mask.sum())

    @property
    def is_valid(self) -> bool:
        return self.matrix is not None

    @property
    def inverse_matrix(self) -> np.ndarray:
        """
        Returns the inverse homography matrix.
        """
        return np.linalg.inv(self.matrix)

class HomographyEstimator:
    """
    Estimates the homography matrix between image coordinates
    and football pitch world coordinates.
    """

    def __init__(
        self,
        method: int = cv2.RANSAC,
        ransac_threshold: float = 5.0,
    ) -> None:

        self.method = method
        self.ransac_threshold = ransac_threshold

    def estimate(
        self,
        detections: PitchDetections,
        pitch: PitchConfiguration,
    ) -> HomographyResult:
        """
        Estimate homography from detected pitch keypoints.
        """

        if not detections.has_minimum_points():
            raise ValueError(
                "At least four keypoints are required to estimate homography."
            )

        image_points = detections.image_points()

        world_points = pitch.world_points_by_ids(
            detections.class_ids()
        )

        matrix, mask = cv2.findHomography(
            srcPoints=image_points,
            dstPoints=world_points,
            method=self.method,
            ransacReprojThreshold=self.ransac_threshold,
        )

        if matrix is None:
            raise RuntimeError(
                "Failed to estimate homography matrix."
            )

        return HomographyResult(
            matrix=matrix,
            mask=mask,
        )