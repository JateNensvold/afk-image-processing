import faiss
from image_processing.load_images import display_image
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

        self.kmeans = faiss.Kmeans(d=self.dimensions,
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)

    def fit_image(self, image: np.ndarray):
        """_summary_

        Args:
            image (np.ndarray): image in BGR format
        """
        self.image = image
        pixels: np.ndarray = np.float32(image.reshape(-1, 3))
        # Filter out all pixels with a sum/value below 1 to ignore masked pixels
        # pylint: disable=unsubscriptable-object
        # filtered_pixels = pixels[pixels[:, 2] >
        #                          1 * np.all(pixels[:, 0:2] == 0, 1), :]
        # print(f"Pixel count {len(filtered_pixels)}/{len(pixels)} - {len(filtered_pixels)/len(pixels)}")
        filtered_pixels = pixels
        if len(filtered_pixels) == 0:
            return np.ndarray([])
        self.fit(filtered_pixels)

    def fit(self, pixels: np.ndarray):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """

        self.kmeans.train(pixels.astype(np.float32))


        import image_processing.globals as GV

        import time

        start_time = time.time()

        for pixel in pixels:
            output = self.kmeans.assign(np.array(pixel).reshape(-1, 3))
        print(f"pixel count: {time.time() - start_time}")
            
            # print(output)
        # print(self.kmeans.centroids)
        # filtered_image = np.zeros(pixels.shape)



        # centroids = self.kmeans.centroids
        # for index, pixel in enumerate(pixels):
        #     new_pixel = self.find_nearest(centroids, pixel, axis=1)
        #     filtered_image[index] = new_pixel
        #     # output = self.kmeans.assign(pixel)
        #     # print(output)
        display_image([self.image, np.reshape(
            filtered_image, self.image.shape)], display=True)

    # def find_nearest(self, array: np.ndarray, cell: int, axis:int):
    #     """_summary_

    #     Args:
    #         array (np.ndarray): _description_
    #         cell (int): _description_

    #     Returns:
    #         _type_: _description_
    #     """

    #     # filtered_pixels = pixels[pixels[:, 2] >
    #     #                          1 * np.all(pixels[:, 0:2] == 0, 1), :]


    #     idx = np.abs(array[0] - cell[0] + array[1]-cell[1] + array[2]-cell[2]).argmin()
    #     print(cell, array.shape, idx)
    #     return array[idx]
