from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        


    def get_clustering_model(self, image):
        # Reshape the image to a 2D array of pixels
        image_2d = image.reshape((-1, 3))

        # preform KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, init = 'k-means++', n_init=1)
        kmeans.fit(image_2d)
        return kmeans
        

    def get_player_color(self, frame, bbox):
        image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        top_half_image = image[:image.shape[0]//2, :]

        # get clusting model
        kmeans = self.get_clustering_model(top_half_image)

        # get the cluster labels for each pixel in the image
        labels = kmeans.labels_

        # Reshape the labels to the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get the player clustrer by checking the corner pixels of the clustered image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = 1 if corner_clusters.count(1) > corner_clusters.count(0) else 0
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
        


    def assign_team_color(self,frame ,player_detection):

        player_colors = []
        for _,player_detection in player_detection.iterrows():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, random_state=0, init = 'k-means++', n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors['team1'] = kmeans.cluster_centers_[0]
        self.team_colors['team2'] = kmeans.cluster_centers_[1]



    def get_player_team(self,frame ,player_bbox , player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id
        return team_id

            