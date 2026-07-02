from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from inference import get_model



@dataclass(frozen=True, slots=True)
class DetectedKeyPoint:
    """
    Represents a detected pitch keypoint in image coordinates.
    """

    class_id: int
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class PitchDetections:
    """
    Collection of detected pitch keypoints for a single frame.
    """

    keypoints: List[DetectedKeyPoint] = field(default_factory=list)

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    @property
    def is_empty(self) -> bool:
        return self.num_keypoints == 0

    def has_minimum_points(self, min_points: int = 4) -> bool:
        return self.num_keypoints >= min_points

    def get(self, class_id: int) -> Optional[DetectedKeyPoint]:
        for kp in self.keypoints:
            if kp.class_id == class_id:
                return kp
        return None

    def class_ids(self) -> List[int]:
        return [kp.class_id for kp in self.keypoints]

    def image_points(self) -> np.ndarray:
        return np.asarray(
            [(kp.x, kp.y) for kp in self.keypoints],
            dtype=np.float32,
        )

    def image_points_by_ids(
        self,
        class_ids: List[int],
    ) -> np.ndarray:

        points = []

        for class_id in class_ids:

            kp = self.get(class_id)

            if kp is not None:
                points.append((kp.x, kp.y))

        return np.asarray(points, dtype=np.float32)



class RoboflowPitchDetector:
    """
    Detect football pitch keypoints using a Roboflow Keypoints model.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        confidence_threshold: float = 0.5,
    ) -> None:

        self.model = get_model(
            model_id=model_id,
            api_key=api_key,
        )

        self.confidence_threshold = confidence_threshold

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Apply preprocessing before inference.
        """

        return frame

    def _inference(
        self,
        frame: np.ndarray,
    ):
        """
        Run Roboflow inference.
        """

        return self.model.infer(frame)[0]

    def _postprocess(
        self,
        prediction,
    ) -> PitchDetections:
        """
        Convert Roboflow output to PitchDetectionResult.
        """

        if not prediction.predictions:
            return PitchDetections()

        prediction = prediction.predictions[0]

        keypoints: List[DetectedKeyPoint] = []

        for kp in prediction.keypoints:

            if kp.confidence < self.confidence_threshold:
                continue

            keypoints.append(
                DetectedKeyPoint(
                    class_id=int(kp.class_id),
                    x=float(kp.x),
                    y=float(kp.y),
                    confidence=float(kp.confidence),
                )
            )

        return PitchDetections(keypoints)

    def detect(
        self,
        frame: np.ndarray,
    ) -> PitchDetections:
        """
        Detect football pitch keypoints from a frame.
        """

        frame = self._preprocess(frame)

        prediction = self._inference(frame)

        return self._postprocess(prediction)