from typing import Iterable

import numpy as np


class DistanceCalculator:
    """
    Utility class for calculating distances on the football pitch.

    Notes
    -----
    World coordinates are expected to be in standard pitch coordinate units.
    Returned distances are in meters.
    """

    def __init__(
        self,
        units_per_meter: float = 1000.0,
    ) -> None:
        self._scale = units_per_meter

    def between(
        self,
        point1: tuple[float, float] | np.ndarray,
        point2: tuple[float, float] | np.ndarray,
    ) -> float:
        """
        Calculate the distance between two world points.

        Parameters
        ----------
        point1 : (x, y)
        point2 : (x, y)

        Returns
        -------
        float
            Distance in meters.
        """

        point1 = np.asarray(point1, dtype=np.float32)
        point2 = np.asarray(point2, dtype=np.float32)

        return float(
            np.linalg.norm(point2 - point1) / self._scale
        )

    def path_length(
        self,
        points: Iterable[tuple[float, float]],
    ) -> float:
        """
        Calculate the total travelled distance.

        Parameters
        ----------
        points :
            Sequence of world coordinates.

        Returns
        -------
        float
            Total distance in meters.
        """

        points = np.asarray(list(points), dtype=np.float32)

        if len(points) < 2:
            return 0.0

        deltas = np.diff(points, axis=0)

        distances = np.linalg.norm(
            deltas,
            axis=1,
        )

        return float(
            distances.sum() / self._scale
        )