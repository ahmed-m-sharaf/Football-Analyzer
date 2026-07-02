from typing import Tuple


def center(
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Returns the center of a bounding box.
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def bottom_center(
    bbox: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Returns the bottom-center point of a bounding box.
    """

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        y2,
    )