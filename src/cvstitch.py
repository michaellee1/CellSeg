# cvstitch.py
# ---------------------------
# Contains the logic for stitching masks.  See class doc for details.

import numpy as np
import cv2

class CVMaskStitcher():
    """
    Implements basic stitching between mask subtiles of semi-uniform size (see constraints below).  
    Initialized with the pixel overlap between masks and the threshold over which an overlap is considered 
    to be one cell, and returns the full set of masks for the passed in rows and cols.
    """
    def __init__(self, overlap=80, threshold=8):
        self.overlap = overlap
        self.threshold = threshold

    # Constraint: Height must be the same between the two masks, and must have width of > OVERLAP.
    # masks1 -> masks2 is left -> right
    def stitch_masks_horizontally(self, masks1, masks2):
        i, j1, N1 = masks1.shape
        _, j2, N2 = masks2.shape
        
        masks_overlap1 = masks1[:, -self.overlap:, :]
        masks_overlap2 = masks2[:, :self.overlap, :]
        
        pad_after_left = j2 - self.overlap
        pad_before_right = j1 - self.overlap

        # squash 3D masks into 2D single plane instance masks, then flatten to 1D arrays
        plane_mask_1, plane_mask_2 = np.zeros(1), np.zeros(1)
        if N1 != 0:
            plane_mask_1 = np.max(np.arange(1,N1+1, dtype=np.uint16)[None,None,:]*masks_overlap1, axis=2).flatten()
        if N2 != 0:
            plane_mask_2 = np.max(np.arange(1,N2+1, dtype=np.uint16)[None,None,:]*masks_overlap2, axis=2).flatten()
        if N1 != 0 and N2 != 0:
            # M is the binary intersection array to capture two mask instances overlap
            M = np.zeros((N1 + 1, N2 + 1))
            np.add.at(M, (plane_mask_1, plane_mask_2), 1)

            del_indices_1 = []
            del_indices_2 = []

            for a in range(1, N1 + 1):
                for b in range(1, N2 + 1):
                    if M[a, b] > self.threshold:
                        if len(np.where(masks1[:,:,a-1])[0]) > len(np.where(masks2[:,:,b-1])[0]):
                            del_indices_2.append(b-1)
                        else:
                            del_indices_1.append(a-1)

            masks1 = masks1[:, :, list(set(np.arange(0, N1)) - set(del_indices_1))]
            masks2 = masks2[:, :, list(set(np.arange(0, N2)) - set(del_indices_2))]
        masks1 = np.pad(masks1,[(0,0),(0,pad_after_left), (0,0)], 'constant')
        masks2 = np.pad(masks2,[(0,0),(pad_before_right,0), (0,0)], 'constant')

        return np.concatenate((masks1, masks2), axis=2)

    # Constraint: Width must be the same between the two masks, and must have height of > OVERLAP.
    # masks1 -> masks2 is top -> bottom
    def stitch_masks_vertically(self, masks1, masks2):
        i1, j, N1 = masks1.shape
        i2, _, N2 = masks2.shape
        
        masks_overlap1 = masks1[-self.overlap:, :, :]
        masks_overlap2 = masks2[:self.overlap, :, :]
        
        pad_below_top = i2 - self.overlap
        pad_above_bottom = i1 - self.overlap

        plane_mask_1, plane_mask_2 = np.zeros(1), np.zeros(1)
        if N1 != 0:
            plane_mask_1 = np.max(np.arange(1,N1+1, dtype=np.uint16)[None,None,:]*masks_overlap1, axis=2).flatten()
        if N2 != 0:
            plane_mask_2 = np.max(np.arange(1,N2+1, dtype=np.uint16)[None,None,:]*masks_overlap2, axis=2).flatten()
        if N1 != 0 and N2 != 0:
            # M is the binary intersection array to capture two mask instances overlap
            M = np.zeros((N1 + 1, N2 + 1))
            np.add.at(M, (plane_mask_1, plane_mask_2), 1)

            del_indices_1 = []
            del_indices_2 = []

            for a in range(1, N1 + 1):
                for b in range(1, N2 + 1):
                    if M[a, b] > self.threshold:
                        if len(np.where(masks1[:,:,a-1])[0]) > len(np.where(masks2[:,:,b-1])[0]):
                            del_indices_2.append(b-1)
                        else:
                            del_indices_1.append(a-1)

            masks1 = masks1[:, :, list(set(np.arange(0, N1)) - set(del_indices_1))]
            masks2 = masks2[:, :, list(set(np.arange(0, N2)) - set(del_indices_2))]
        masks1 = np.pad(masks1,[(0,pad_below_top),(0,0), (0,0)], 'constant')
        masks2 = np.pad(masks2,[(pad_above_bottom,0),(0,0), (0,0)], 'constant')

        return np.concatenate((masks1, masks2), axis=2)
    
    def stitch_masks(self, masks, nrows, ncols):
        assert(len(masks) == nrows * ncols)
        
        for i in range(len(masks)):
            masks[i] = self.remove_small_cells(masks[i])

        horizontal_strips = []
        
        # Create horizontal strips
        for i in range(0, len(masks), ncols):
            strip = masks[i]
            
            for a in range(i+1, i+ncols):
                strip = self.stitch_masks_horizontally(strip, masks[a])
            horizontal_strips.append(strip)
        
        # Stitch horizontal strips
        full_mask = horizontal_strips[0]
        for j in range(1, nrows):
            full_mask = self.stitch_masks_vertically(full_mask, horizontal_strips[j])
        
        return full_mask

    # Remove any cells smaller than the defined threshold.
    def remove_small_cells(self, masks):
        i, j, n_masks = masks.shape
        channel_counts = np.zeros((n_masks + 1), dtype='uint16')
        plane_mask = np.zeros(1, dtype='uint16')
        if n_masks != 0:
            plane_mask = np.max(np.arange(1,n_masks+1, dtype=np.uint16)[None,None,:]*masks, axis=2).flatten()
        np.add.at(channel_counts, plane_mask, 1)
        keep_indices = np.where(channel_counts[1:] > self.threshold)
        return masks[:, :, keep_indices].squeeze()

