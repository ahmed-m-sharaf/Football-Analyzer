from typing import Optional
import cv2
import numpy as np

from .homography import HomographyResult


class CoordinateTransformer:
    """
    Transform coordinates between image space and football pitch
    world space.
    """

    def __init__(self) -> None:

        self._homography: Optional[HomographyResult] = None

    @property
    def is_initialized(self) -> bool:
        return self._homography is not None

    @property
    def homography(self) -> HomographyResult:

        if not self.is_initialized:
            raise RuntimeError(
                "Transformer has not been initialized."
            )

        return self._homography

    def update(
        self,
        homography: HomographyResult,
    ) -> None:
        """
        Update the current homography.
        """

        self._homography = homography

    def reset(self) -> None:
        """
        Reset transformer.
        """

        self._homography = None

    def image_to_world(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Transform a single image point into world coordinates.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "Transformer has not been initialized."
            )

        points = np.asarray(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            points,
            self.homography.matrix,
        )

        x, y = transformed[0, 0]

        return float(x), float(y)

    def image_to_world_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Transform multiple image points.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "Transformer has not been initialized."
            )

        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.homography.matrix,
        ).reshape(-1, 2)

    def world_to_image(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Transform a single world point into image coordinates.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "Transformer has not been initialized."
            )

        points = np.asarray(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            points,
            self.homography.inverse_matrix,
        )

        x, y = transformed[0, 0]

        return float(x), float(y)

    def world_to_image_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Transform multiple world points.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "Transformer has not been initialized."
            )

        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.homography.inverse_matrix,
        ).reshape(-1, 2)