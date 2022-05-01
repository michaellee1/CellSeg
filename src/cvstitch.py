# cvstitch.py
# ---------------------------
# Contains the logic for stitching masks.  See class doc for details.

import numpy as np
import cv2
import sys
import time
from math import ceil
import matplotlib.pyplot as plt
import warnings
import pandas as pd

class CVMaskStitcher():
    """
    Implements basic stitching between mask subtiles of semi-uniform size (see constraints below).  
    Initialized with the pixel overlap between masks and the threshold over which an overlap is considered 
    to be one cell, and returns the full set of masks for the passed in rows and cols.
    """
    def __init__(self, overlap=80, threshold=8):
        self.overlap = overlap
        self.threshold = threshold
        self.memory_max = 15

    #reindexes masks for final stitching so no two masks have same id number        
    def renumber_masks(self, masks):
        prev_max = 0

        for i, crop in enumerate(masks):
            if np.any(crop.astype(bool)):
                newcrop = np.copy(crop)
                newcrop[crop > 0] += prev_max
                prev_max += len(np.unique(newcrop)) - 1
                masks[i] = newcrop

        return masks
    
    def flat_to_expanded(self, planemasks):
        mask_arr_h, mask_arr_w = planemasks.shape
        num_masks = len(np.unique(planemasks)) - 1
        expanded_mask_arr = np.zeros((mask_arr_h, mask_arr_w, num_masks + 1), dtype = bool)
        flat_mask_arr = planemasks.flatten()
        oldmaskids = np.sort(np.unique(flat_mask_arr))
        newmaskids = np.arange(len(oldmaskids))
        maskmap = dict(zip(oldmaskids, newmaskids))
        flat_mask_arr = np.vectorize(maskmap.get)(flat_mask_arr)
        row_indices = np.repeat(np.arange(mask_arr_h), mask_arr_w)
        col_indices = np.array(list(np.arange(mask_arr_w)) * mask_arr_h)
        index_array = [row_indices, col_indices, flat_mask_arr]
        flat_index_array = np.ravel_multi_index(
            index_array,
            expanded_mask_arr.shape)
        np.ravel(expanded_mask_arr)[flat_index_array] = True
        expanded_mask_arr = expanded_mask_arr[:, :, 1:]
        
        return expanded_mask_arr

    def stitch_masks(self, masks, nrows, ncols):
        #if there was no cropping for segmentation, return the segmented image
        if len(masks) == 1: 
            return self.flat_to_expanded(masks[0])
        
        assert(len(masks) == nrows * ncols)
        
        masks = self.renumber_masks(masks)

        #overlap is the amount of overlap between crops, which is set to 80 pixels as default

        #first remove cells under a certain size to get rid of artifacts
        print(f"Removing masks with area less than {self.threshold} pixels.")
        mask_ids = []
        mask_sizes = []
        for i in range(len(masks)):
            masks[i], sizes, ids = self.remove_small_cells(masks[i])
            mask_sizes.extend(sizes)
            mask_ids.extend(ids)
            
        size_dict = dict(zip(mask_ids, mask_sizes))

        del mask_ids, mask_sizes

        h, w = masks[0].shape
        
        strip_w, strip_h = 0, 0
        
        if ncols == 1:
            strip_w = w
        else:
            strip_w = (w * 2) + (w + int(self.overlap / 2)) * (ncols - 2) - self.overlap * (ncols - 1)
        
        if nrows == 1:
            strip_h = h
        else:
            strip_h = (h * 2) + (h + int(self.overlap / 2)) * (nrows - 2) - self.overlap * (nrows - 1)

        nlayers = 4

        expanded_mask_arr = np.zeros((strip_h, strip_w, nlayers), dtype = np.int32)

        layer_to_populate_1 = 0
        curr_left = 0
        curr_top = 0
        cropshape = (0,0)

        for i in range(0, len(masks), ncols):
            curr_left = 0
            currrow = int(i / ncols)

            if currrow % 2 == 0:
                layer_to_populate_1 = 0
            else:
                layer_to_populate_1 = 2

            for j in range(i, i + ncols):
                currmasks = masks[j]
                currcol = j - i
                cropshape = currmasks.shape
                layer_to_populate = layer_to_populate_1 + (currcol % 2)
                expanded_mask_arr[curr_top:(curr_top + cropshape[0]), curr_left:(curr_left + cropshape[1]), layer_to_populate] = currmasks
                curr_left += cropshape[1] - self.overlap
                
            curr_top += cropshape[0] - self.overlap
               
                
        mask_overlaps = np.sum(expanded_mask_arr > 0, axis = 2) > 1
        mask_overlaps_compress = np.zeros((4, np.sum(mask_overlaps)), dtype = np.int32)

        for i in range(nlayers):
            mask_overlaps_compress[i, :] = expanded_mask_arr[:, :, i][mask_overlaps]

        mask_conflicts = np.unique(mask_overlaps_compress, axis = 1)

        del mask_overlaps, mask_overlaps_compress

        num_conflicts = mask_conflicts.shape[1]
        
        #now compare mask sizes of overlapping masks, only keeping the largest mask in each conflict 
        #and removing all other masks

        masks_to_rem = []

        for i in range(num_conflicts):
            idlist = mask_conflicts[:, i]
            idlist = [f for f in idlist if f > 0]
            currsizes = [size_dict.get(maskid) for maskid in idlist]
            largest_mask_idx = np.argmax(np.array(currsizes))
            largest_mask = idlist[largest_mask_idx]
            remlist = [f for f in idlist if f != largest_mask]
            masks_to_rem.extend(remlist)

        masks_to_rem = list(set(masks_to_rem)) #get just unique ids in list
        masklocs = np.isin(expanded_mask_arr, masks_to_rem)
        expanded_mask_arr[masklocs] = 0
        full_mask_arr = np.sum(expanded_mask_arr, axis = 2)
        num_masks = len(np.unique(full_mask_arr)) - 1
        print("Showing stitched masks")
        plt.imshow(full_mask_arr > 0)
        plt.show()
        #warn users if the number of masks discovered in the image will crash the program
        mask_arr_h, mask_arr_w = full_mask_arr.shape
        #expected_memory_gb = mask_arr_h * mask_arr_w * (num_masks + 1) / 1e9
        #if expected_memory_gb > self.memory_max:
        #    warnings.warn(f"The number of masks found and dimensions of image will cause program to exceed {self.memory_max + 1} GB memory usage. Consider splitting image into subtiles and segmenting them individually.", UserWarning)
        
        #expanded_masks = self.flat_to_expanded(full_mask_arr)
        
        #renumber the indices before returning array
        indices = np.nonzero(full_mask_arr)
        values = full_mask_arr[indices]
        valframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"x", 1:"y", 2:"idx"})
        valframe = valframe.sort_values(by = 'idx')
        num_idx = valframe['idx'].value_counts(sort = False).sort_index().tolist()
        valframe['idx'] = np.repeat(np.arange(1, num_masks + 1), num_idx)
        valarray = valframe.to_numpy()
        full_mask_arr[valarray[:, 0], valarray[:, 1]] = valarray[:, 2]

        #return expanded_masks
        return full_mask_arr

    # Remove any cells smaller than the defined threshold.
    def remove_small_cells(self, mask):
        mask_id, sizes = np.unique(mask, return_counts = True)
        keep_indices = list(sizes > self.threshold)
        for currid, keep_id in zip(mask_id, keep_indices):
            if not keep_id:
                mask[mask == currid] = 0

        return mask, list(sizes[keep_indices]), list(mask_id[keep_indices])
