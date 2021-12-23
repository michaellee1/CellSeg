# cvmask.py
# ---------------------------
# Wrapper class for masks.  See class doc for details.

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial import distance
from operator import itemgetter
from skimage.measure import find_contours
from skimage.morphology import disk, dilation
from scipy.ndimage.morphology import binary_dilation
import pandas as pd
import sys
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

IMAGEJ_BAND_WIDTH = 200
EIGHT_BIT_MAX = 255

class CVMask():
    '''
    Provides a class that wraps around a numpy array representing masks out of the CellVision model.
    The class provides functions to grow, remove overlaps (nearest neighbors), and export to various
    formats.  All methods that change the masks modify the masks stored in the .masks property
    '''
    def __init__(self, flatmasks):
        self.masks = None
        self.flatmasks = flatmasks
        self.centroids = None

    def n_instances(self):
        return len(np.unique(self.flatmasks)) - 1

    def update_adjacency_value(self, adjacency_matrix, original, neighbor):
        border = False

        if original != 0 and original != neighbor:
            border = True
            if neighbor != 0:
                adjacency_matrix[int(original - 1), int(neighbor - 1)] += 1
        return border

    def update_adjacency_matrix(self, plane_mask_flattened, width, height, adjacency_matrix, index):
        mod_value_width = index % width
        origin_mask = plane_mask_flattened[index]
        left, right, up, down = False, False, False, False

        if (mod_value_width != 0):
            left = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-1])
        if (mod_value_width != width - 1):
            right = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+1])
        if (index >= width):
            up = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index-width])
        if (index <= len(plane_mask_flattened) - 1 - width):
            down = self.update_adjacency_value(adjacency_matrix, origin_mask, plane_mask_flattened[index+width])
        
        if (left or right or up or down):
            adjacency_matrix[int(origin_mask - 1), int(origin_mask-1)] += 1

    def compute_channel_means_sums_compensated(self, image):
        height, width, n_channels = image.shape
        mask_height, mask_width = self.flatmasks.shape
        n_masks = len(np.unique(self.flatmasks)) - 1
        channel_sums = np.zeros((n_masks, n_channels))
        channel_counts = np.zeros((n_masks, n_channels))
        if n_masks == 0:
            return channel_sums, channel_sums, channel_counts

        squashed_image = np.reshape(image, (height*width, n_channels))
        
        #masklocs = np.nonzero(self.flatmasks)
        #plane_mask = np.zeros((mask_height, mask_width), dtype = np.uint32)
        #plane_mask[masklocs[0], masklocs[1]] = masklocs[2] + 1
        #plane_mask = plane_mask.flatten()
        plane_mask = self.flatmasks.flatten()
        
        adjacency_matrix = np.zeros((n_masks, n_masks))
        for i in range(len(plane_mask)):
            self.update_adjacency_matrix(plane_mask, mask_width, mask_height, adjacency_matrix, i)
            
            mask_val = plane_mask[i] - 1
            if mask_val != -1:
                channel_sums[mask_val.astype(np.int32)] += squashed_image[i]
                channel_counts[mask_val.astype(np.int32)] += 1
        
        
        # Normalize adjacency matrix
        for i in range(n_masks):
            adjacency_matrix[i] = adjacency_matrix[i] / (max(adjacency_matrix[i, i], 1) * 2)
            adjacency_matrix[i, i] = 1
        
        means = np.true_divide(channel_sums, channel_counts, out=np.zeros_like(channel_sums, dtype='float'), where=channel_counts!=0)
        results = lstsq(adjacency_matrix, means, overwrite_a=True, overwrite_b=False)
        compensated_means = np.maximum(results[0], np.zeros((1,1)))        

        return compensated_means, means, channel_counts[:,0]

    def compute_centroids(self):
        masks = self.flatmasks
        num_masks = len(np.unique(masks)) - 1
        indices = np.where(masks != 0)
        values = masks[indices[0], indices[1]]

        maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"x", 1:"y", 2:"id"})
        centroids = maskframe.groupby('id').agg({'x': 'mean', 'y': 'mean'}).to_records(index = False).tolist()
        
        self.centroids = centroids
        
        
    def compute_boundbox(self):
        masks = self.flatmasks
        num_masks = len(np.unique(masks)) - 1
        indices = np.where(masks != 0)
        values = masks[indices[0], indices[1]]

        maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"y", 1:"x", 2:"id"})
        self.bb_mins = maskframe.groupby('id').agg({'y': 'min', 'x': 'min'}).to_records(index = False).tolist()
        self.bb_maxes = maskframe.groupby('id').agg({'y': 'max', 'x': 'max'}).to_records(index = False).tolist()
    
    def absolute_centroids(self, tile_row, tile_col):
        y_offset = self.flatmasks.shape[0] * (tile_row - 1)
        x_offset = self.flatmasks.shape[1] * (tile_col - 1)

        offsets = [y_offset, x_offset]

        centroids = self.centroids
        if not centroids:
            return centroids
        
        absolutes = [(cent[0] + offsets[0], cent[1] + offsets[1]) for cent in centroids]
        
        absolutes = np.array(absolutes)

        return absolutes
    
    def applyXYoffset(masks,offset_vector):

        for i in range(masks.shape[2]):
            masks[0,:,i] += offset_vector[0]
            masks[1,:,i] += offset_vector[1]
        return masks
            
    def remove_overlaps_nearest_neighbors(self, masks):
        final_masks = np.max(masks, axis = 2)
        centroids = self.centroids
        collisions = np.nonzero(np.sum(masks > 0, axis = 2) > 1)
        collision_masks = masks[collisions]
        collision_index = np.nonzero(collision_masks)
        collision_masks = collision_masks[collision_index]
        collision_frame = pd.DataFrame(np.transpose(np.array([collision_index[0], collision_masks]))).rename(columns = {0:"collis_idx", 1:"mask_id"})
        grouped_frame = collision_frame.groupby('collis_idx')
        for collis_idx, group in grouped_frame:
            collis_pos = np.expand_dims(np.array([collisions[0][collis_idx], collisions[1][collis_idx]]), axis = 0)
            prevval = final_masks[collis_pos[0,0], collis_pos[0,1]]
            mask_ids = list(group['mask_id'])
            curr_centroids = np.array([centroids[mask_id - 1] for mask_id in mask_ids])
            dists = cdist(curr_centroids, collis_pos)
            closest_mask = mask_ids[np.argmin(dists)]
            final_masks[collis_pos[0,0], collis_pos[0,1]] = closest_mask
        
        return final_masks
             
    def grow_masks(self, growth, method = 'Standard', num_neighbors = 30):
        assert method in ['Standard', 'Sequential']
        
        masks = self.flatmasks
        num_masks = len(np.unique(masks)) - 1
        
        if method == 'Standard':
            print("Standard growth selected")
            masks = self.flatmasks
            num_masks = len(np.unique(masks)) - 1
            indices = np.where(masks != 0)
            values = masks[indices[0], indices[1]]

            maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"x", 1:"y", 2:"id"})
            cent_array = maskframe.groupby('id').agg({'x': 'mean', 'y': 'mean'}).to_numpy()
            connectivity_matrix = kneighbors_graph(cent_array, num_neighbors).toarray() * np.arange(1, num_masks + 1)
            connectivity_matrix = connectivity_matrix.astype(int)
            labels = {}
            for n in range(num_masks):
                connections = list(connectivity_matrix[n, :])
                connections.remove(0)
                layers_used = [labels[i] for i in connections if i in labels]
                layers_used.sort()
                currlayer = 0
                for layer in layers_used:
                    if currlayer != layer: 
                        break
                    currlayer += 1
                labels[n + 1] = currlayer

            possible_layers = len(list(set(labels.values())))
            label_frame = pd.DataFrame(list(labels.items()), columns = ["maskid", "layer"])
            image_h, image_w = masks.shape
            expanded_masks = np.zeros((image_h, image_w, possible_layers), dtype = np.uint32)

            grouped_frame = label_frame.groupby('layer')
            for layer, group in grouped_frame:
                currids = list(group['maskid'])
                masklocs = np.isin(masks, currids)
                expanded_masks[masklocs, layer] = masks[masklocs]

            dilation_mask = disk(1)
            grown_masks = np.copy(expanded_masks)
            for _ in range(growth):
                for i in range(possible_layers):
                    grown_masks[:, :, i] = dilation(grown_masks[:, :, i], dilation_mask)
            self.flatmasks = self.remove_overlaps_nearest_neighbors(grown_masks)
                
        elif method == 'Sequential':
            print("Sequential growth selected")
            Y, X = masks.shape
            struc = disk(1)
            for _ in range(growth):
                for i in range(num_masks):
                    mins = self.bb_mins[i]
                    maxes = self.bb_maxes[i]
                    minY, minX, maxY, maxX = mins[0] - 3*growth, mins[1] - 3*growth, maxes[0] + 3*growth, maxes[1] + 3*growth
                    if minX < 0: minX = 0
                    if minY < 0: minY = 0
                    if maxX >= X: maxX = X - 1
                    if maxY >= Y: maxY = Y - 1

                    currreg = masks[minY:maxY, minX:maxX]
                    mask_snippet = (currreg == i + 1)
                    full_snippet = currreg > 0
                    other_masks_snippet = full_snippet ^ mask_snippet
                    dilated_mask = binary_dilation(mask_snippet, struc)
                    final_update = (dilated_mask ^ full_snippet) ^ other_masks_snippet

                    #f, axarr = plt.subplots(1, 5)
                    #plt.imshow(mask_snippet)
                    #axarr[0].imshow(mask_snippet)
                    #axarr[1].imshow(full_snippet)
                    #axarr[2].imshow(other_masks_snippet)
                    #axarr[3].imshow(dilated_mask)
                    #axarr[4].imshow(final_update)
                    #plt.show()

                    pix_to_update = np.nonzero(final_update)

                    pix_X = np.array([min(j + minX, X) for j in pix_to_update[1]])
                    pix_Y = np.array([min(j + minY, Y) for j in pix_to_update[0]])

                    masks[pix_Y, pix_X] = i + 1

            self.flatmasks = masks

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
