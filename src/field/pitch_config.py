from dataclasses import dataclass
from typing import List, Tuple

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

    Stores:
        - Pitch dimensions
        - World coordinates of all pitch landmarks
        - Connections between landmarks
    """

    def __init__(self) -> None:

        config = SoccerPitchConfiguration()

        # Pitch Dimension
        self.length: float = config.length
        self.width: float = config.width

        self.penalty_box_length: float = config.penalty_box_length
        self.penalty_box_width: float = config.penalty_box_width

        self.goal_box_length: float = config.goal_box_length
        self.goal_box_width: float = config.goal_box_width

        self.center_circle_radius: float = config.centre_circle_radius
        self.penalty_spot_distance: float = config.penalty_spot_distance

        # World Coordinates
        self.world_coordinates: List[Tuple[float, float]] = config.vertices

        labels: List[str] = config.labels

        self._points: List[PitchPoint] = [
            PitchPoint(
                id=i,
                x=x,
                y=y,
                label=labels[i],
            )
            for i, (x, y) in enumerate(self.world_coordinates)
        ]

        self._point_map = {
            point.id: point
            for point in self._points
        }

        # Pitch Connections
        self._edges: List[Tuple[int, int]] = config.edges

    @property
    def points(self) -> List[PitchPoint]:
        return self._points.copy()

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._edges.copy()

    @property
    def num_points(self) -> int:
        return len(self._points)

    @property
    def dimensions(self) -> Tuple[float, float]:
        return self.length, self.width

    def is_valid_point(self, class_id: int) -> bool:
        return class_id in self._point_map

    def get_point(self, class_id: int) -> PitchPoint:
        if not self.is_valid_point(class_id):
            raise ValueError(f"Invalid pitch point id: {class_id}")

        return self._point_map[class_id]

    def get_points(self, class_ids: List[int]) -> List[PitchPoint]:
        return [self.get_point(i) for i in class_ids]

    def get_all_points(self) -> List[PitchPoint]:
        return self.points