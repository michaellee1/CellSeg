import os
from src.cvsegmenter import CVSegmenter
from src.cvstitch import CVMaskStitcher
from src.cvmask import CVMask
from src import cvutils
from src import cvvisualize
from src import fcswrite
from cvconfig import CVConfig
import skimage       
import numpy as np
import pandas as pd
import tensorflow as tf

# -------------------
# START
# -------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
cf = CVConfig()

print('Initializing CVSegmenter at', cf.DIRECTORY_PATH)
print('Picking channel', cf.NUCLEAR_CHANNEL_NAME, 'from',
      len(cf.CHANNEL_NAMES), 'total to segment on')
print('Channel names:')
print(cf.CHANNEL_NAMES)
print("Working with images of shape:", cf.SHAPE)

stitcher = CVMaskStitcher(overlap=cf.OVERLAP)
segmenter = CVSegmenter(
    cf.SHAPE,
    cf.MODEL_PATH,
    cf.OVERLAP,
    cf.INCREASE_FACTOR
)
# ------------------
# HELPERS (Refactor)
# ------------------

def boost_image(image, boost_factor):
    EIGHT_BIT_MAX = 255

    max_mask = np.full(image.shape, EIGHT_BIT_MAX)
    boosted = np.rint(np.minimum(
        image * boost_factor, max_mask)).astype('uint8')
    return skimage.img_as_ubyte(boosted)


def get_channel_index(text):
    if text not in cf.CHANNEL_NAMES:
        raise NameError('Couldn\'t find {} channel'.format(text))

    return np.where(cf.CHANNEL_NAMES == text)[0][0]

# ---------------
# OUTPUT HANDLING
# ---------------

# for growth in [1,4,2,3]:
growth = cf.GROWTH_PIXELS
rows, cols = None, None
dataframe_list = []

if cf.OUTPUT_METHOD not in ['imagej_text_file', 'statistics', 'visual_image_output', 'visual_overlay_output', 'all']:
    raise NameError(
            'Output method is not supported.  Check the OUTPUT_METHOD variable in cvconfig.py.')

for filename in cf.FILENAMES:
    path = os.path.join(cf.DIRECTORY_PATH, filename)
    image = skimage.external.tifffile.imread(path).reshape(cf.SHAPE)
    nuclear_index = get_channel_index(cf.NUCLEAR_CHANNEL_NAME)
    nuclear_image = np.expand_dims(skimage.img_as_ubyte(image)[nuclear_index], axis=2)
    nuclear_image = boost_image(nuclear_image[:, :, [0, 0, 0]], cf.BOOST)

    print('\nSegmenting with CellVision:', filename)
    masks, rows, cols = segmenter.segment_image(nuclear_image)

    print('Stitching:', filename)
    stitched_mask = CVMask(stitcher.stitch_masks(masks, rows, cols))
    instances = stitched_mask.n_instances()
    print(instances, 'cell masks found by segmenter')
    if instances == 0:
        print('No cells found in', filename, ', skipping to next')
        continue

    print('Growing cells by', growth, 'pixels:', filename)
    stitched_mask.grow_by(growth)
    print('Removing overlaps by nearest neighbor:', filename)
    stitched_mask.remove_overlaps_nearest_neighbors()

    if cf.OUTPUT_METHOD == 'imagej_text_file':
        print('Sort into strips and outputting:', filename)
        new_path = os.path.join(
            cf.IMAGEJ_OUTPUT_PATH, (filename[:-4] + '-coords.txt'))
        stitched_mask.sort_into_strips()
        stitched_mask.output_to_file(new_path)

    if cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all':
        print('Calculating statistics:', filename)
        reg, tile_row, tile_col, tile_z = cvutils.extract_tile_information(
            filename)
        channel_means, size = stitched_mask.compute_channel_means_sums(image)
        centroids = stitched_mask.compute_centroids()
        absolutes = stitched_mask.absolute_centroids(tile_row, tile_col)

        if centroids.size != 0:
            metadata_list = np.array([reg, tile_row, tile_col, tile_z])
            metadata = np.broadcast_to(
                metadata_list, (stitched_mask.n_instances(), len(metadata_list)))

            print(metadata.shape, centroids.shape, absolutes.shape, size[:, None].shape, channel_means.shape)
            semi_dataframe = np.concatenate(
                [metadata, centroids, absolutes, size[:, None], channel_means], axis=1)
            dataframe_list.append(semi_dataframe)

    if cf.OUTPUT_METHOD == 'visual_image_output' or cf.OUTPUT_METHOD == 'all':
        print('Creating visual output saved to', cf.VISUAL_OUTPUT_PATH)
        new_path = os.path.join(cf.VISUAL_OUTPUT_PATH, filename[:-4]) + 'visual_growth' + str(growth)
        figsize = (cf.SHAPE[2] // 25, cf.SHAPE[1] // 25)
        cvvisualize.generate_instances_and_save(
            new_path + '.png', nuclear_image, stitched_mask.masks, figsize=figsize)
    
    if cf.OUTPUT_METHOD == 'visual_overlay_output' or cf.OUTPUT_METHOD == 'all':
        print('Creating visual overlay output saved to', cf.VISUAL_OUTPUT_PATH)
        new_path = os.path.join(cf.VISUAL_OUTPUT_PATH, filename[:-4]) + 'growth' + str(growth) + 'mask.tif'
        cvvisualize.generate_masks_and_save(
            new_path, nuclear_image, stitched_mask.masks)        

if cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all':
    full_df_array = np.concatenate(dataframe_list, axis=0)
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
    columns = descriptive_labels + list(cf.CHANNEL_NAMES)
    dataframe = pd.DataFrame(full_df_array, columns=columns)

    path = os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, filename[:-4] +'statistics_growth' + str(growth))
    dataframe.to_csv(path + '.csv')
    # Output to .fcs file
    fcswrite.write_fcs(path + '.fcs', columns, full_df_array)
