# walkthrough_utils.py
# ---------------------------
# Utilities for jupyter notebook walkthrough.  See class doc for details.

from src import cvutils
from src import fcswrite
from PIL import Image
import pandas as pd
import skimage
import os
import numpy as np

def compute_boost(imagepath, read_method, is_codex_output, shape, n_dims, nuc_chan_name, chan_names, autoboost_perc = 99.98):
    image = np.array(read_method(imagepath))
    ext = imagepath.split('.')[-1]
    if 'tif' in ext:
        if n_dims == 4:
            image = np.transpose(image, (2, 3, 0, 1))
        elif n_dims == 3:
            image = np.transpose(image, (1, 2, 0))
    image = image.reshape(shape)
    nuclear_index = None
    if 'tif' in ext:
        nuclear_index = cvutils.get_channel_index(nuc_chan_name, chan_names)
    nuclear_image = cvutils.get_nuclear_image(n_dims-1, image, nuclear_index=nuclear_index)

    print('Using auto boosting - may be inaccurate for empty or noisy images.', '\n')
    image_max = np.percentile(nuclear_image, autoboost_perc)
    boost = cvutils.EIGHT_BIT_MAX / image_max
    print('Boosting with value of', boost, ', check that this makes sense.')
    return boost

def compute_stats(grown_masks, cur_im_name, image, IS_CODEX_OUTPUT, CHANNEL_NAMES, SHOULD_COMPENSATE, QUANTIFICATION_OUTPUT_PATH, CSV_OR_FCS, growth):
    
    if CSV_OR_FCS not in ['csv', 'fcs']:
        raise NameError(
            "CSV_OR_FCS parameter must be one of csv or fcs")
    
    print('Calculating statistics:', cur_im_name)
    reg, tile_row, tile_col, tile_z = 0, 1, 1, 0
    if IS_CODEX_OUTPUT:
        reg, tile_row, tile_col, tile_z = cvutils.extract_tile_information(
            cur_im_name)
    channel_means, size = None, None
    
    channel_means_comp, channel_means_uncomp, size = grown_masks.compute_channel_means_sums_compensated(image)
    
    centroids = grown_masks.centroids
    absolutes = grown_masks.absolute_centroids(tile_row, tile_col)
    semi_dataframe_comp = 1
    semi_dataframe = 1
    if centroids:
        metadata_list = np.array([reg, tile_row, tile_col, tile_z])
        metadata = np.broadcast_to(
            metadata_list, (grown_masks.n_instances(), len(metadata_list)))

        semi_dataframe = np.concatenate(
            [metadata, np.array(centroids), absolutes, size[:, None], channel_means_uncomp], axis=1)
        semi_dataframe_comp = np.concatenate(
            [metadata, np.array(centroids), absolutes, size[:, None], channel_means_comp], axis=1)

    descriptive_labels = [
        'Reg',
        'Tile Row',
        'Tile Col',
        'Tile Z',
        'In-Tile Y',
        'In-Tile X',
        'Absolute Y',
        'Absolute X',
        'Cell Size'
    ] 

    # Output to CSV
    ext = cur_im_name.split('.')[-1]
    if not IS_CODEX_OUTPUT:
        regname = cur_im_name.replace("." + cur_im_name.split(".")[-1], "")
        if not 'tif' in ext:
            CHANNEL_NAMES = ['single-channel']
            n_channels = image.shape[2]
            if n_channels == 3:
                CHANNEL_NAMES = ['Red', 'Green', 'Blue']
    columns = descriptive_labels + [s for s in CHANNEL_NAMES]
    dataframe = pd.DataFrame()
    path = ''
    regname = cur_im_name.split("_")[0]
    if SHOULD_COMPENSATE:
        dataframe = pd.DataFrame(semi_dataframe_comp, columns=columns)
        path = os.path.join(QUANTIFICATION_OUTPUT_PATH, regname + '_statistics_growth_' + str(growth)+'_comp')
    else:
        dataframe = pd.DataFrame(semi_dataframe, columns=columns)
        path = os.path.join(QUANTIFICATION_OUTPUT_PATH, regname + '_statistics_growth_' + str(growth)+'_uncomp')
        
    if CSV_OR_FCS == 'csv':
        if os.path.exists(path+'.csv'):
            dataframe.to_csv(path + '.csv',mode='a',header=False)
        else:
            dataframe.to_csv(path + '.csv')
    elif CSV_OR_FCS == 'fcs':
        fcswrite.write_fcs(path + '.fcs', columns, dataframe)
