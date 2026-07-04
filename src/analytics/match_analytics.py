import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from src.utils.bbox import center, bottom_center
from src.analytics.distance import DistanceCalculator

class MatchAnalytics:
    """
    Computes match-level and player-level analytics including speed,
    distance covered, and team possession.
    """

    def __init__(
        self,
        fps: float = 30.0,
        possession_threshold_meters: float = 2.0,
        speed_window_size: int = 15,
        max_valid_step_meters: float = 12.0,  # Max distance a player can cover in 1 frame (~432 km/h, filters jumps)
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self.possession_threshold = possession_threshold_meters
        self.speed_window_size = speed_window_size
        self.max_valid_step = max_valid_step_meters

        # Initialize distance calculator with a scale of 100.0 (centimeters to meters)
        self.distance_calc = DistanceCalculator(millimeters_per_meter=100.0)

        # Player tracking dictionaries
        # tracker_id -> list of world positions
        self.player_positions: Dict[int, List[Tuple[float, float]]] = {}
        # tracker_id -> total distance covered in meters
        self.player_distances: Dict[int, float] = {}
        # tracker_id -> deque of recent speeds for smoothing
        self.player_speeds_history: Dict[int, deque] = {}
        # tracker_id -> current smoothed speed in km/h
        self.player_current_speeds: Dict[int, float] = {}

        # Team possession statistics
        self.possession_frames: Dict[int, int] = {1: 0, 2: 0}
        self.total_possession_frames = 0
        self.last_possession_team: Optional[int] = None

    def update(
        self,
        players: List[dict],
        ball: List[dict],
        transformer,
        frame_idx: int,
    ) -> Dict[str, any]:
        """
        Update analytics for the current frame.

        Parameters
        ----------
        players : List[dict]
            List of detected/tracked players in the current frame.
        ball : List[dict]
            List of detected ball objects in the current frame (expected 0 or 1).
        transformer : CoordinateTransformer
            Transformer to convert image coordinates to world coordinates.
        frame_idx : int
            Current frame index.

        Returns
        -------
        Dict[str, any]
            Calculated metrics for the current frame.
        """
        # 1. Update player positions, distance covered, and speed
        for player in players:
            tracker_id = player["tracker_id"]
            bbox = player["bbox"]
            class_id = player["class_id"]  # 1: goalkeeper, 2: player, 3: referee
            
            # Skip referees for distance and speed calculations
            if class_id == 3:
                continue

            # Calculate bottom center in image coordinates
            img_pos = bottom_center(bbox)
            
            # Convert to world coordinates
            is_world = False
            if transformer is not None and transformer.is_initialized:
                try:
                    world_pos = transformer.image_to_world(img_pos)
                    is_world = True
                except Exception:
                    world_pos = img_pos
            else:
                world_pos = img_pos

            # Track positions
            if tracker_id not in self.player_positions:
                self.player_positions[tracker_id] = []
                self.player_distances[tracker_id] = 0.0
                self.player_speeds_history[tracker_id] = deque(maxlen=self.speed_window_size)
                self.player_current_speeds[tracker_id] = 0.0

            # Calculate movement since last frame
            if len(self.player_positions[tracker_id]) > 0:
                prev_pos = self.player_positions[tracker_id][-1]
                if is_world:
                    step_dist = self.distance_calc.between(prev_pos, world_pos)
                else:
                    # In pixel space, assume 0.03 meters per pixel (approximate scale)
                    pixel_dist = np.linalg.norm(np.array(world_pos) - np.array(prev_pos))
                    step_dist = float(pixel_dist * 0.03)
                
                # Check for tracking jumps/re-identification leaps
                # Maximum physical running speed of a human is ~40 km/h = 11.11 m/s.
                # Maximum step distance in 1 frame is: 11.11 * dt
                max_step = 11.11 * self.dt
                # Add tolerance for coordinate noise/jitter
                max_step_with_tolerance = max_step * 1.5
                
                if step_dist < max_step_with_tolerance:
                    # Filter out micro-movements/jitter when players are stationary
                    if step_dist < 0.015:
                        step_dist = 0.0
                    
                    self.player_distances[tracker_id] += step_dist
                    
                    # Calculate speed
                    speed_ms = step_dist / self.dt
                    speed_kmh = min(speed_ms * 3.6, 40.0) # Cap at realistic maximum human speed (40 km/h)
                    self.player_speeds_history[tracker_id].append(speed_kmh)
                else:
                    # Ignore anomalous tracking jump/ID-swap steps
                    pass
            
            # Record current position
            self.player_positions[tracker_id].append(world_pos)

            # Compute smoothed speed
            speeds = self.player_speeds_history[tracker_id]
            if len(speeds) > 0:
                self.player_current_speeds[tracker_id] = float(np.mean(speeds))
            else:
                self.player_current_speeds[tracker_id] = 0.0

        # 2. Update ball possession
        possession_event = None
        if len(ball) > 0:
            ball_bbox = ball[0]["bbox"]
            ball_img_pos = center(ball_bbox)
            
            is_world_ball = False
            if transformer is not None and transformer.is_initialized:
                try:
                    ball_world_pos = transformer.image_to_world(ball_img_pos)
                    is_world_ball = True
                except Exception:
                    ball_world_pos = ball_img_pos
            else:
                ball_world_pos = ball_img_pos
                
            # Find the closest team player (excluding referees)
            closest_player = None
            min_dist = float("inf")
            
            for player in players:
                if player["class_id"] == 3:  # Skip referees
                    continue
                
                tracker_id = player["tracker_id"]
                if tracker_id in self.player_positions and len(self.player_positions[tracker_id]) > 0:
                    player_world_pos = self.player_positions[tracker_id][-1]
                    
                    if is_world_ball:
                        dist = self.distance_calc.between(player_world_pos, ball_world_pos)
                    else:
                        pixel_dist = np.linalg.norm(np.array(player_world_pos) - np.array(ball_world_pos))
                        dist = float(pixel_dist * 0.03)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = player
            
            # Assign possession if within threshold (2.0 meters)
            if closest_player is not None and min_dist <= self.possession_threshold:
                team_id = closest_player.get("team_id")
                if team_id in [1, 2]:
                    self.possession_frames[team_id] += 1
                    self.total_possession_frames += 1
                    self.last_possession_team = team_id
                    possession_event = team_id
            else:
                # Ball is free but last team is still considered in control for continuity
                if self.last_possession_team is not None:
                    self.possession_frames[self.last_possession_team] += 1
                    self.total_possession_frames += 1
                    possession_event = self.last_possession_team
        else:
            # No ball detected, carry forward previous possession if exists
            if self.last_possession_team is not None:
                self.possession_frames[self.last_possession_team] += 1
                self.total_possession_frames += 1
                possession_event = self.last_possession_team

        # Calculate percentages
        possession_percentages = {1: 50.0, 2: 50.0}
        if self.total_possession_frames > 0:
            possession_percentages[1] = (self.possession_frames[1] / self.total_possession_frames) * 100.0
            possession_percentages[2] = (self.possession_frames[2] / self.total_possession_frames) * 100.0

        return {
            "possession_team": possession_event,
            "possession_percentages": possession_percentages,
            "player_distances": self.player_distances.copy(),
            "player_speeds": self.player_current_speeds.copy(),
        }
