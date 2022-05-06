import faiss
import numpy as np


class PixelKMeans:
    """_summary_
    """

    def __init__(self, dimensions=3, n_clusters=5, n_init=1, max_iter=200):
        """_summary_

        Args:
            n_clusters (int, optional): _description_. Defaults to 8.
            n_init (int, optional): _description_. Defaults to 10.
            max_iter (int, optional): _description_. Defaults to 300.
        """
        self.dimensions = dimensions
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

    def fit_image(self, image: np.ndarray):
        """_summary_

        Args:
            image (np.ndarray): image in BGR format
        """
        pixels: np.ndarray = np.float32(image.reshape(-1, 3))
        # Filter out all pixels with a sum/value below 1 to ignore masked pixels
        # pylint: disable=unsubscriptable-object
        filtered_pixels = pixels[pixels[:, 2] >
                                 1 * np.all(pixels[:, 0:2] == 0, 1), :]
        if len(filtered_pixels) == 0:
            return np.ndarray([])
        return self.fit(filtered_pixels)

    def fit(self, pixels: np.ndarray):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        kmeans = faiss.Kmeans(d=self.dimensions,
                              k=self.n_clusters,
                              niter=self.max_iter,
                              nredo=self.n_init)
        kmeans.train(pixels.astype(np.float32))
        return kmeans.centroids
