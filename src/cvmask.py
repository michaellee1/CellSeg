import numpy as np
from scipy.spatial import distance
from operator import itemgetter
from skimage.measure import find_contours

IMAGEJ_BAND_WIDTH = 200
EIGHT_BIT_MAX = 255

class CVMask():
    '''
    Provides a class that wraps around a numpy array representing masks out of the CellVision model.
    The class provides functions to grow, remove overlaps (nearest neighbors), and export to various
    formats.  All methods that change the masks modify the masks stored in the .masks property
    '''
    def __init__(self, masks):
        self.masks = masks
        self.centroids = None

    @staticmethod
    def bounding_box(Y, X, max_y, max_x, growth):
        minX = np.maximum(0, np.min(X) - growth)
        minY = np.maximum(0, np.min(Y) - growth)
        maxY = np.minimum(max_y, np.max(Y) + growth)
        maxX = np.minimum(max_x, np.max(X) + growth)

        return (minX, minY, maxX, maxY)

    @staticmethod
    def get_centroid(Y, X):
        return (np.mean(Y), np.mean(X))

    @staticmethod
    def expand_snippet(snippet, pixels):
        y_len,x_len = snippet.shape
        output = snippet.copy()
        for _ in range(pixels):
            for y in range(y_len):
                for x in range(x_len):
                    if (y > 0        and snippet[y-1,x]) or \
                    (y < y_len - 1 and snippet[y+1,x]) or \
                    (x > 0        and snippet[y,x-1]) or \
                    (x < x_len - 1 and snippet[y,x+1]): output[y,x] = True
            snippet = output.copy()
        return output

    def n_instances(self):
        if len(self.masks.shape) < 3:
            return 0
        return self.masks.shape[2]

    def compute_channel_means_sums(self, image):
        n_channels, height, width = image.shape
        mask_height, mask_width, n_masks = self.masks.shape
        channel_sums = np.zeros((n_masks, n_channels))
        channel_counts = np.zeros((n_masks, n_channels))
        if n_masks == 0:
            return channel_sums, channel_counts

        squashed_image = np.reshape(image.transpose(1,2,0), (height*width, n_channels))

        plane_mask = np.max(np.arange(1,n_masks+1, dtype=np.uint16)[None,None,:]*self.masks, axis=2).flatten()

        for i in range(squashed_image.shape[0]):
            mask_val = plane_mask[i] - 1
            if mask_val != -1:
                channel_sums[mask_val] += squashed_image[i]
                channel_counts[mask_val] += 1

        means = np.true_divide(channel_sums, channel_counts, out=np.zeros_like(channel_sums, dtype='float'), where=channel_counts!=0)
        return means, channel_counts[:,0]

    def compute_centroids(self):
        if self.centroids is None:
            self.centroids = []
            for i in range(self.n_instances()):
                mask = self.masks[:, :, i]
                coords = np.where(mask)
                centroid = self.get_centroid(coords[0], coords[1])
                self.centroids.append(centroid)

        return np.array(self.centroids)
    
    def absolute_centroids(self, tile_row, tile_col):
        y_offset = self.masks.shape[0] * (tile_row - 1)
        x_offset = self.masks.shape[1] * (tile_col - 1)

        offsets = np.array([y_offset, x_offset])

        centroids = self.compute_centroids()
        if centroids.size == 0:
            return centroids

        absolutes = centroids + offsets

        return absolutes

    def grow_by(self, growth):
        Y, X, N = self.masks.shape

        self.centroids = []
        self.bb_mins = []
        self.bb_maxes = []

        import timeit
        start = timeit.default_timer()
        for i in range(N):
            mask = self.masks[:, :, i]

            stop = timeit.default_timer()
            print('Time First: ', stop - start) 

            coords = np.where(mask)

            stop = timeit.default_timer()
            print('Time: ', stop - start)

            minX, minY, maxX, maxY = self.bounding_box(coords[0], coords[1], Y-1, X-1, growth)
            self.bb_mins.append((minX, minY))
            self.bb_maxes.append((maxX, maxY))

            stop = timeit.default_timer()
            print('Time: ', stop - start)
            
            centroid = self.get_centroid(coords[0], coords[1])
            self.centroids.append(centroid)

            stop = timeit.default_timer()
            print('Time: ', stop - start)
            
            snippet = mask[minY:maxY,minX:maxX]
            new_snippet = self.expand_snippet(snippet, growth)

            stop = timeit.default_timer()
            print('Time: ', stop - start)
            
            mask[minY:maxY,minX:maxX] = new_snippet

            stop = timeit.default_timer()
            print('Time: ', stop - start)

    def remove_overlaps_nearest_neighbors(self):
        Y, X, N = self.masks.shape

        collisions = []
        for y in range(Y):
            for x in range(X):
                pixel_masks = np.where(self.masks[y, x, :])[0]
                if len(pixel_masks) == 2:
                    collisions.append(pixel_masks)

        for collision in collisions:
            c1, c2 = collision[0], collision[1]
            minX, minY = np.minimum(np.array(self.bb_mins[c1]), np.array(self.bb_mins[c2]))
            maxX, maxY = np.maximum(np.array(self.bb_maxes[c1]), np.array(self.bb_maxes[c2]))
            c_pixels = np.where(self.masks[minY:maxY,minX:maxX,c1] & self.masks[minY:maxY,minX:maxX,c2])
            Y_collision = c_pixels[0]
            X_collision = c_pixels[1]

            for i in range(len(Y_collision)):
                y_offset = minY + Y_collision[i]
                x_offset = minX + X_collision[i]
                
                distance_to_c0 = distance.euclidean((x_offset, y_offset), self.centroids[c1])
                distance_to_c1 = distance.euclidean((x_offset, y_offset), self.centroids[c2])
                
                if distance_to_c0 > distance_to_c1:
                    self.masks[y_offset, x_offset, c1] = False
                else:
                    self.masks[y_offset, x_offset, c2] = False

    def sort_into_strips(self):
        N = self.n_instances()
        unsorted = []
        
        for n in range(N):
            mask_coords = np.where(self.masks[:,:,n])
            if (len(mask_coords[0]) > 0):
                y = mask_coords[0][0]
                x = mask_coords[1][0] // IMAGEJ_BAND_WIDTH
                unsorted.append((x, y, n))

        sorted_masks = sorted(unsorted, key=itemgetter(0,1))
        self.masks = self.masks[:, :, [x[2] for x in sorted_masks]]

    def output_to_file(self, file_path):
        N = self.n_instances()
        vertex_array = []

        for i in range(N):
            # Mask
            mask = self.masks[:, :, i]

            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                vertex_array.append(verts)

        X, Y = [], []
        for i in range(len(vertex_array)):
            x, y = zip(*(vertex_array[i]))
            X.append(x)
            Y.append(y)
            
        # Needs to be awkwardly written into file because Fiji doesn't have extensions like numpy or pickle
        with open(file_path, "w") as f:
            for i in range(len(X)):
                line = ""
                for j in range(len(X[i])):
                    line += str(X[i][j]) + " "
                line = line.strip() + ","
                for k in range(len(Y[i])):
                    line += str(Y[i][k]) + " "
                line = line.strip() + "\n"
                f.write(line)

    # def compute_statistics(self, image):

