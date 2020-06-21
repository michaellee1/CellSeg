# main.py
# ---------------------------
# main.py connects segmentation, stitching, and output into a single pipeline.  It prints metadata about
# the run, and then initializes a segmenter and stitcher.  Looping over all image files in the directory,
# each image is segmented, stitched, grown, and overlaps resolved.  The data is concatenated if outputting
# as quantifications, and outputted per file for other output methods.  This file can be run by itself by
# invoking python main.py or the main function imported.

import os
from src.cvsegmenter import CVSegmenter
from src.cvstitch import CVMaskStitcher
from src.cvmask import CVMask
from src import cvutils
from src import cvvisualize
from src import fcswrite
from cvconfig import CVConfig
from PIL import Image
import skimage       
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    cf = CVConfig()

    print('Initializing CSSegmenter at', cf.DIRECTORY_PATH)
    if cf.IS_CODEX_OUTPUT:
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
        cf.INCREASE_FACTOR,
        cf.THRESHOLD
    )

    growth = cf.GROWTH_PIXELS
    rows, cols = None, None
    dataframe_regs = defaultdict(list)
    columns = []
    path=''

    if cf.OUTPUT_METHOD not in ['imagej_text_file', 'statistics', 'visual_image_output', 'visual_overlay_output', 'all']:
        raise NameError(
                'Output method is not supported.  Check the OUTPUT_METHOD variable in cvconfig.py.')
    print("Checking previous segmentation progress...")
    progress_table = cf.PROGRESS_TABLE
    print("These tiles already segmented: ")
    print(progress_table)
    cf.FILENAMES = [item for item in cf.FILENAMES if item not in progress_table]
    cf.FILENAMES.sort()
    
    count = 0

    for filename in cf.FILENAMES:
        count += 1

        path = os.path.join(cf.DIRECTORY_PATH, filename)
        image = np.array(cf.READ_METHOD(path))
        if cf.IS_CODEX_OUTPUT:
            image = np.transpose(image, (2, 3, 0, 1))
        image = image.reshape(cf.SHAPE)
        nuclear_index = None
        if cf.N_DIMS == 4:
            nuclear_index = cvutils.get_channel_index(cf.NUCLEAR_CHANNEL_NAME, cf.CHANNEL_NAMES)
        nuclear_image = cvutils.get_nuclear_image(cf.N_DIMS-1, image, nuclear_index=nuclear_index)

        if cf.BOOST == 'auto':
            print('Using auto boosting - may be inaccurate for empty or noisy images.')
            image_max = np.percentile(nuclear_image, cf.AUTOBOOST_PERCENTILE)
            cf.BOOST = cvutils.EIGHT_BIT_MAX / image_max
            print('Boosting with value of', cf.BOOST, ', check that this makes sense.')

        nuclear_image = cvutils.boost_image(nuclear_image, cf.BOOST)

        print('\nSegmenting with CellSeg:', filename)
        masks, rows, cols = segmenter.segment_image(nuclear_image)

        print('Stitching:', filename)
        stitched_mask = CVMask(stitcher.stitch_masks(masks, rows, cols))

        instances = stitched_mask.n_instances()
        print(instances, 'cell masks found by segmenter')
        if instances == 0:
            print('No cells found in', filename, ', skipping to next')
            continue

        print('Growing cells by', growth, 'pixels:', filename)
        if cf.USE_SEQUENTIAL_GROWTH:
            print('Sequential growth selected')
            stitched_mask.grow_by(0)
            print('Removing overlaps by nearest neighbor:', filename)
            stitched_mask.remove_overlaps_nearest_neighbors()
            stitched_mask.new_grow_by(growth)
            #print('Applying XY offset', filename)
        else:
            #stitched_mask.binarydilate(growth)
            #stitched_mask.remove_conflicts_nn()
            stitched_mask.grow_by(growth)
            print('Removing overlaps by nearest neighbor:', filename)
            stitched_mask.remove_overlaps_nearest_neighbors()
        #    print('Applying XY offset', filename)
        
        #record masks as flattened array
        stitched_mask.flatten_masks()
        

        if not os.path.exists(cf.IMAGEJ_OUTPUT_PATH):
            os.makedirs(cf.IMAGEJ_OUTPUT_PATH)
        if not os.path.exists(cf.VISUAL_OUTPUT_PATH):
            os.makedirs(cf.VISUAL_OUTPUT_PATH)
        if not os.path.exists(cf.QUANTIFICATION_OUTPUT_PATH):
            os.makedirs(cf.QUANTIFICATION_OUTPUT_PATH)

        if cf.OUTPUT_METHOD == 'imagej_text_file':
            print('Sort into strips and outputting:', filename)
            new_path = os.path.join(
                cf.IMAGEJ_OUTPUT_PATH, (filename[:-4] + '-coords.txt'))
            stitched_mask.sort_into_strips()
            stitched_mask.output_to_file(new_path)

        if cf.OUTPUT_METHOD == 'visual_image_output' or cf.OUTPUT_METHOD == 'all':
            print('Creating visual output saved to', cf.VISUAL_OUTPUT_PATH)
            new_path = os.path.join(cf.VISUAL_OUTPUT_PATH, filename[:-4]) + 'visual_growth' + str(growth)
            figsize = (cf.SHAPE[1] // 25, cf.SHAPE[0] // 25)
            cvvisualize.generate_instances_and_save(
                new_path + '.png', nuclear_image, stitched_mask.masks[1:,1:,:], figsize=figsize)
        
        if cf.OUTPUT_METHOD == 'visual_overlay_output' or cf.OUTPUT_METHOD == 'all':
            print('Creating visual overlay output saved to', cf.VISUAL_OUTPUT_PATH)
            new_path = os.path.join(cf.VISUAL_OUTPUT_PATH, filename[:-4]) + 'growth' + str(growth) + 'mask.tif'
            cvvisualize.generate_masks_and_save(new_path, nuclear_image, stitched_mask.masks[1:,1:,:])

        if cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all':
            print('Calculating statistics:', filename)
            reg, tile_row, tile_col, tile_z = 0, 1, 1, 0
            if cf.IS_CODEX_OUTPUT:
                reg, tile_row, tile_col, tile_z = cvutils.extract_tile_information(
                    filename)
            channel_means, size = None, None

            channel_means_comp, channel_means_uncomp, size = stitched_mask.compute_channel_means_sums_compensated(image)

            centroids = stitched_mask.compute_centroids()
            absolutes = stitched_mask.absolute_centroids(tile_row, tile_col)
            semi_dataframe_comp = 1
            if centroids.size != 0:
                metadata_list = np.array([reg, tile_row, tile_col, tile_z])
                metadata = np.broadcast_to(
                    metadata_list, (stitched_mask.n_instances(), len(metadata_list)))

                semi_dataframe = np.concatenate(
                    [metadata, centroids, absolutes, size[:, None], channel_means_uncomp], axis=1)
                semi_dataframe_comp = np.concatenate(
                    [metadata, centroids, absolutes, size[:, None], channel_means_comp], axis=1)

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
            if not cf.IS_CODEX_OUTPUT:
                cf.CHANNEL_NAMES = ['single-channel']
                n_channels = cf.SHAPE[2]
                if n_channels == 3:
                    cf.CHANNEL_NAMES = ['Red', 'Green', 'Blue']
            columns = descriptive_labels + [s for s in cf.CHANNEL_NAMES]
            dataframe = None
            path = ''
            regname = filename.split("_")[0]
            if cf.SHOULD_COMPENSATE:
                dataframe = pd.DataFrame(semi_dataframe_comp, columns=columns)
                path = os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, regname + '_statistics_growth' + str(growth)+'_comp')
            else:
                dataframe = pd.DataFrame(semi_dataframe, columns=columns)
                path = os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, regname + '_statistics_growth' + str(growth)+'_uncomp')
            if os.path.exists(path+'.csv'):
                dataframe.to_csv(path + '.csv',mode='a',header=False)
            else:
                dataframe.to_csv(path + '.csv')
            
        #save intermediate progress in case of mid-run crash
        with open(cf.PROGRESS_TABLE_PATH, "a") as myfile:
            myfile.write(filename + "\n")
            
    #duplicate existing csv files in fcs format at the end of run
    if cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all':
        print("Duplicating existing csv files in fcs format")
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
        columns = descriptive_labels + [s for s in cf.CHANNEL_NAMES]
        filenames = os.listdir(cf.QUANTIFICATION_OUTPUT_PATH)
        for filename in filenames:
            path = os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,filename)
            dataframe = pd.read_csv(path,index_col=0)
            path = path.replace('.csv','')
            fcswrite.write_fcs(path + '.fcs', columns, dataframe)
    print("Segmentation Completed")
if __name__ == "__main__":
    main()
