import cv2
import numpy as np
from typing import Dict, List, Tuple

class MatchVisualizer:
    """
    Handles drawing premium quality overlays on match video frames:
    - Team-colored circles at players' feet (semi-transparent).
    - Player IDs, speeds, and distances.
    - Top HUD for ball possession percentages.
    - Ball indicator tracker.
    """

    def __init__(
        self,
        team1_color_bgr: Tuple[int, int, int] = (255, 50, 50),     # Default: Bright Red
        team2_color_bgr: Tuple[int, int, int] = (50, 50, 255),     # Default: Bright Blue
        referee_color_bgr: Tuple[int, int, int] = (0, 240, 240),   # Default: Yellow/Gold
        goalkeeper_color_bgr: Tuple[int, int, int] = (50, 200, 50), # Default: Green
        ball_color_bgr: Tuple[int, int, int] = (0, 165, 255),      # Default: Orange
    ) -> None:
        self.default_colors = {
            "team1": team1_color_bgr,
            "team2": team2_color_bgr,
            "referee": referee_color_bgr,
            "goalkeeper": goalkeeper_color_bgr,
            "ball": ball_color_bgr,
        }

    def draw_player_circles(
        self,
        frame: np.ndarray,
        players: List[dict],
        player_teams: Dict[int, int],
        team_colors_kmeans: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Draws a semi-transparent ellipse at the feet of each player,
        colored according to their team.
        """
        overlay = frame.copy()

        # Resolve team colors (either use KMeans cluster colors or fallback defaults)
        t1_color = self.default_colors["team1"]
        t2_color = self.default_colors["team2"]

        if "team1" in team_colors_kmeans:
            # Convert KMeans RGB/BGR values to integer tuples
            t1_color = tuple(map(int, team_colors_kmeans["team1"]))
        if "team2" in team_colors_kmeans:
            t2_color = tuple(map(int, team_colors_kmeans["team2"]))

        for player in players:
            bbox = player["bbox"]
            tracker_id = player["tracker_id"]
            class_id = player["class_id"]

            x1, y1, x2, y2 = bbox
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            # Determine color
            if class_id == 3:  # Referee
                color = self.default_colors["referee"]
            elif class_id == 1:  # Goalkeeper
                color = self.default_colors["goalkeeper"]
            else:  # Standard Player (class_id == 2)
                team = player_teams.get(tracker_id, 1)
                color = t1_color if team == 1 else t2_color

            # Draw semi-transparent ellipse at the player's feet
            cv2.ellipse(
                overlay,
                center=(foot_x, foot_y),
                axes=(35, 14),
                angle=0.0,
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            # Add an outer border for definition
            cv2.ellipse(
                frame,
                center=(foot_x, foot_y),
                axes=(35, 14),
                angle=0.0,
                startAngle=0,
                endAngle=360,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Blend overlay into frame for transparency
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_player_tags(
        self,
        frame: np.ndarray,
        players: List[dict],
        player_teams: Dict[int, int],
        player_speeds: Dict[int, float],
        player_distances: Dict[int, float],
    ) -> np.ndarray:
        """
        Draws tags above players showing their ID, speed, and distance covered.
        """
        for player in players:
            bbox = player["bbox"]
            tracker_id = player["tracker_id"]
            class_id = player["class_id"]

            # Skip referee tags or show simple ones
            if class_id == 3:
                continue

            x1, y1, x2, y2 = bbox
            top_x = int((x1 + x2) / 2)
            top_y = int(y1) - 10

            speed = player_speeds.get(tracker_id, 0.0)
            dist = player_distances.get(tracker_id, 0.0)

            # Build tag strings
            id_text = f"ID:{tracker_id}"
            stats_text = f"{speed:.1f}km/h | {dist:.1f}m"

            # Draw backgrounds for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(stats_text, font, font_scale, thickness)

            max_w = max(w1, w2)
            box_x1 = top_x - max_w // 2 - 6
            box_y1 = top_y - h1 - h2 - 12
            box_x2 = top_x + max_w // 2 + 6
            box_y2 = top_y + 4

            # Draw background box
            cv2.rectangle(
                frame,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (15, 15, 15),
                thickness=-1,
            )
            # Draw white border
            cv2.rectangle(
                frame,
                (box_x1, box_y1),
                (box_x2, box_y2),
                (220, 220, 220),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Draw text
            cv2.putText(
                frame,
                id_text,
                (top_x - w1 // 2, box_y1 + h1 + 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                stats_text,
                (top_x - w2 // 2, box_y2 - 6),
                font,
                font_scale,
                (0, 220, 220),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return frame

    def draw_ball_marker(
        self,
        frame: np.ndarray,
        ball: List[dict],
    ) -> np.ndarray:
        """
        Draws a tracking pointer triangle above the ball.
        """
        if len(ball) == 0:
            return frame

        bbox = ball[0]["bbox"]
        x1, y1, x2, y2 = bbox
        bx = int((x1 + x2) / 2)
        by = int(y1) - 6

        # Draw a small orange triangle pointing down at the ball
        pts = np.array([
            [bx, by],
            [bx - 10, by - 15],
            [bx + 10, by - 15]
        ], dtype=np.int32)

        cv2.drawContours(
            frame,
            [pts],
            -1,
            self.default_colors["ball"],
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.drawContours(
            frame,
            [pts],
            -1,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return frame

    def draw_possession_hud(
        self,
        frame: np.ndarray,
        possession_percentages: Dict[int, float],
    ) -> np.ndarray:
        """
        Draws a premium possession banner at the top of the frame.
        """
        h, w, _ = frame.shape
        hud_w = 400
        hud_h = 75
        hud_x = (w - hud_w) // 2
        hud_y = 20

        # Semi-transparent dark background for HUD
        hud_overlay = frame.copy()
        cv2.rectangle(
            hud_overlay,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (20, 20, 20),
            thickness=-1,
        )
        
        # Blend overlay
        cv2.addWeighted(hud_overlay, 0.75, frame, 0.25, 0, frame)

        # Draw outer HUD border
        cv2.rectangle(
            frame,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (100, 100, 100),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        p1 = possession_percentages.get(1, 50.0)
        p2 = possession_percentages.get(2, 50.0)

        # Draw Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_text = "BALL POSSESSION"
        (tw, th), _ = cv2.getTextSize(title_text, font, 0.45, 1)
        cv2.putText(
            frame,
            title_text,
            (hud_x + (hud_w - tw) // 2, hud_y + 18),
            font,
            0.45,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA,
        )

        # Draw Percentage Text
        p1_text = f"TEAM 1: {p1:.1f}%"
        p2_text = f"{p2:.1f}% :TEAM 2"
        
        cv2.putText(
            frame,
            p1_text,
            (hud_x + 15, hud_y + 40),
            font,
            0.45,
            (255, 100, 100),
            1,
            lineType=cv2.LINE_AA,
        )
        (p2_w, _), _ = cv2.getTextSize(p2_text, font, 0.45, 1)
        cv2.putText(
            frame,
            p2_text,
            (hud_x + hud_w - p2_w - 15, hud_y + 40),
            font,
            0.45,
            (100, 100, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # Draw Possession Bar
        bar_x = hud_x + 15
        bar_y = hud_y + 52
        bar_w = hud_w - 30
        bar_h = 10

        # Draw Team 1 Possession Bar (Red)
        t1_width = int(bar_w * (p1 / 100.0))
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + t1_width, bar_y + bar_h),
            (80, 80, 220), # Red-ish (in BGR)
            thickness=-1,
        )
        # Draw Team 2 Possession Bar (Blue)
        cv2.rectangle(
            frame,
            (bar_x + t1_width, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (220, 80, 80), # Blue-ish (in BGR)
            thickness=-1,
        )
        # Bar Divider line
        if 0 < t1_width < bar_w:
            cv2.line(
                frame,
                (bar_x + t1_width, bar_y),
                (bar_x + t1_width, bar_y + bar_h),
                (255, 255, 255),
                1,
            )

        return frame
