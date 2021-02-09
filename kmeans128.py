import numpy as np
from PIL import Image
from numpy.linalg import norm
from pathlib import Path
import os
from colormath.color_objects import LabColor, AdobeRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color


class KMeansMLFinal:
    """
    CS74/174 Final Exam Spring 2020
    KMeans Clustering Applications
    """

    def __init__(self, filename, k=8, max_its=300):
        self.num_centroids = k
        self.max_its = max_its

        self.image, self.image_height, self.image_width = self.load_image(filename)
        self.centroids = self.initialize_centroids()
        self.labels = None

    def load_image(self, filename):
        """
        Converts image into a flattened numpy array
        where each row contains R,G,B color information.
        :param filename: string path to image
        :return: (n, 3) dimensional ndarray
        """
        image = Image.open(filename)
        pixel_np = np.asarray(image)
        return np.reshape(pixel_np, (image.width * image.height, 3)), image.height, image.width

    def reconstruct_image(self):
        """
        Recolor each pixel in the image with the color of the centroid of the cluster it belongs to.
        """
        pixel_centroid = np.array([list(self.centroids[label]) for label in self.labels]).astype("uint8")
        pixel_centroids_reshaped = np.reshape(pixel_centroid, (self.image_height, self.image_width, 3), "C")
        return Image.fromarray(pixel_centroids_reshaped)

    def initialize_centroids(self):
        """
        Initialize random color centroids
        :return: (num_centroids, 3) dimension ndarray where each row is a centroid.
        """
        # randomly pick num_centroids number of colors
        centroid_arr = np.floor(256 * np.random.random_sample((self.num_centroids, 3)))
        return centroid_arr

    def update_centroids(self, labels):
        """
        Computes new centroids as mean color of cluster for each cluster.
        :param labels: (n, ) dimension ndarray where each entry is the closest centroid.
        :return: (self.num_centroids, 3) dimension ndarray where each row represents a centroid.
        """
        new_centroids = np.zeros((self.num_centroids, 3))
        for i in range(self.num_centroids):
            clust = self.image[labels==i]
            if clust.shape[0] > 0:
                mean = np.mean(clust, axis=0)
                new_centroids[i] = mean
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def distance(self, centroids):
        """
        Calculate Euclidean distance of each point to each column
        :param centroids: (self.num_centroids, 3) dimensional ndarray
        :return: (n, self.num_centroids) dimensional ndarray
                where each element indicates the distance of the ith pixel (column) to the jth centroid (row).
        """
        dist_array = np.zeros((self.image_width * self.image_height, self.num_centroids))
        pre_dist = np.tile(self.image, (1, self.num_centroids))
        # pre_dist = np.tile(self.image, (1, self.num_centroids)) - centroids.flatten()
        for i in range(self.image.shape[0]):
            im_color = convert_color(AdobeRGBColor(self.image[i,0], self.image[i,1], self.image[i,2]), LabColor)
            for j in range(self.num_centroids):
                cent_color = convert_color(AdobeRGBColor(centroids[j,0], centroids[j,1], centroids[j,2]), LabColor)
                dist_array[i, j] = delta_e_cie2000(im_color, cent_color)
            # dist_array[:,i] = norm(pre_dist[:,3 * i:3 * (i + 1)], axis=1).reshape((1,self.image_width * self.image_height))
        return dist_array

    def find_nearest_cluster(self, distances):
        """
        Given the matrix of distances (returned from distance method) of each pixel to each centroid,
        find the cluster with the minimum distance for each pixel.
        :param distances: (n, self.num_centroids) dimensional ndarray
        :return: (n, ) dimensional ndarray  where each entry represents the closest cluster for each pixel.
        """
        closest = np.argmin(distances, axis=1)
        return closest

    def fit(self):
        """
        Implement KMeans Clustering Here.
        Please make sure to store your final labels and centroids in self.labels and self.centroids
            * self.labels -- (n, ) shape ndarray where each element is an int
                             indicating which cluster a pixel belongs to
            * self.centroids -- (num_centroids, 3) dimensional ndarray where each row is a centroid (color).
        :return:
        """
        l_hist = []
        c_hist = []
        for i in range(self.max_its):
            old_centroids = self.centroids
            self.labels = self.find_nearest_cluster(self.distance(self.centroids))
            self.centroids = self.update_centroids(self.labels)
            im = self.reconstruct_image()
            im_name = './input/k_' + str(self.num_centroids) + '/im' + str(i) + '.png'
            im_path = Path(im_name)
            im.save(im_path, "png")
            diff = norm(old_centroids - self.centroids)
            if(diff < 1):
                break

# k_range = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]
k_range = [128]
for i in k_range:
    newdir = "./input/k_" + str(i)
    if not os.path.isdir(newdir):
        os.makedirs(newdir)
    model = KMeansMLFinal("./input/baker.jpg", k=i)
    model.fit()
