import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sports.configs.soccer import SoccerPitchConfiguration


@dataclass(frozen=True, slots=True)
class PitchPoint:
    """
    Represents a single football pitch landmark.
    """

    id: int
    x: float
    y: float
    label: str


class PitchConfiguration:
    """
    Represents the geometry of a football pitch.
    """

    def __init__(self) -> None:

        config = SoccerPitchConfiguration()


        self.length: float = config.length
        self.width: float = config.width

        self.penalty_box_length: float = config.penalty_box_length
        self.penalty_box_width: float = config.penalty_box_width

        self.goal_box_length: float = config.goal_box_length
        self.goal_box_width: float = config.goal_box_width

        self.center_circle_radius: float = config.centre_circle_radius
        self.penalty_spot_distance: float = config.penalty_spot_distance

        labels: List[str] = config.labels

        self._points: List[PitchPoint] = [
            PitchPoint(
                id=i,
                x=x,
                y=y,
                label=labels[i],
            )
            for i, (x, y) in enumerate(config.vertices)
        ]

        self._point_map: Dict[int, PitchPoint] = {
            point.id: point
            for point in self._points
        }

        self._world_points = np.asarray(
            [(point.x, point.y) for point in self._points],
            dtype=np.float32,
        )


        self._edges: List[Tuple[int, int]] = config.edges


    @property
    def points(self) -> List[PitchPoint]:
        return self._points.copy()

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._edges.copy()

    @property
    def world_points(self) -> np.ndarray:
        return self._world_points.copy()

    @property
    def num_points(self) -> int:
        return len(self._points)

    @property
    def dimensions(self) -> Tuple[float, float]:
        return self.length, self.width

    @property
    def center(self) -> Tuple[float, float]:
        return (
            self.length / 2,
            self.width / 2,
        )


    def is_valid_point(self, class_id: int) -> bool:
        return class_id in self._point_map

    def get_point(self, class_id: int) -> PitchPoint:
        if not self.is_valid_point(class_id):
            raise KeyError(f"Pitch point {class_id} does not exist.")

        return self._point_map[class_id]

    def get_points(
        self,
        class_ids: List[int],
    ) -> List[PitchPoint]:
        return [self.get_point(class_id) for class_id in class_ids]

    def world_points_by_ids(
        self,
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Returns world coordinates corresponding to the given class ids.
        """

        points = self.get_points(class_ids)

        return np.asarray(
            [(point.x, point.y) for point in points],
            dtype=np.float32,
        )