from dataclasses import dataclass

import cv2
import numpy as np

from .homography import HomographyResult

class CoordinateTransformer:
    """
    Transforms coordinates between image space and pitch world space.
    """

    def __init__(
        self,
        homography: HomographyResult,
    ) -> None:

        if not homography.is_valid:
            raise ValueError("Invalid homography matrix.")

        self._homography = homography

    def image_to_world(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Transform a single image point to world coordinates.
        """

        points = np.asarray(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            points,
            self._homography.matrix,
        )

        x, y = transformed[0, 0]

        return float(x), float(y)
    
    def image_to_world_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Transform multiple image points.
        Shape: (N,2)
        """

        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self._homography.matrix,
        ).reshape(-1, 2)
    
    def world_to_image(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Transform a single world point to image coordinates.
        """

        points = np.asarray(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            points,
            self._homography.inverse_matrix,
        )

        x, y = transformed[0, 0]

        return float(x), float(y)

    def world_to_image_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Transform multiple world points.
        Shape: (N,2)
        """

        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self._homography.inverse_matrix,
        ).reshape(-1, 2)