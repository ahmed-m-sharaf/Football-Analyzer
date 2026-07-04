import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        # Reshape the image to a 2D array of pixels
        image_2d = image.reshape((-1, 3))

        # perform KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++', n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Convert float coordinates to integers and clip to frame dimensions
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2]))
        y2 = min(frame.shape[0], int(bbox[3]))

        # Handle empty or invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0], dtype=np.float32)

        image = frame[y1:y2, x1:x2]

        h, w, _ = image.shape
        top_half_image = image[:max(1, h // 2), :]

        if top_half_image.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)

        # get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # get the cluster labels for each pixel in the image
        labels = kmeans.labels_

        # Reshape the labels to the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get the player cluster by checking the corner pixels of the clustered image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = 1 if corner_clusters.count(1) > corner_clusters.count(0) else 0
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if not player_colors:
            # Fallback if no players detected
            self.team_colors['team1'] = np.array([255, 0, 0])
            self.team_colors['team2'] = np.array([0, 0, 255])
            return

        kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++', n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors['team1'] = kmeans.cluster_centers_[0]
        self.team_colors['team2'] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id, class_id=2, player_teams=None, frame_players=None):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            # If team colors have not been assigned yet, return default team
            return 1

        if class_id == 1:  # Goalkeeper
            # Assign goalkeeper to the team that is closest to them on average
            gk_center = ((player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2)
            
            team1_dists = []
            team2_dists = []
            
            if frame_players is not None and player_teams is not None:
                for p in frame_players:
                    if p["class_id"] == 2:  # Outfield players
                        p_id = p["tracker_id"]
                        t_id = player_teams.get(p_id)
                        if t_id is not None:
                            p_bbox = p["bbox"]
                            p_center = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
                            dist = np.linalg.norm(np.array(gk_center) - np.array(p_center))
                            if t_id == 1:
                                team1_dists.append(dist)
                            elif t_id == 2:
                                team2_dists.append(dist)
            
            if not team1_dists and not team2_dists:
                team_id = 1
            elif not team1_dists:
                team_id = 2
            elif not team2_dists:
                team_id = 1
            else:
                team_id = 1 if np.mean(team1_dists) < np.mean(team2_dists) else 2
        else:  # Standard Player
            player_color = self.get_player_color(frame, player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1

        self.player_team_dict[player_id] = team_id
        return team_id
