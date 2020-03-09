
from src import cvutils
from PIL import Image
import pandas as pd
import skimage
import os

class CVConfig():
    '''
    Define your constants below.

    IS_CODEX_OUTPUT - CODEX output files have special filenames that allow outputs to contain more metadata about absolute positions, regs, and other things.  
    Set this to false if not using the filename convention.  Follow the naming convention in the install/run page on the CellVision website to output this metadata for non-CODEX images.
    DIRECTORY_PATH - directory that contains your .tif bestFocus images (usually the bestFocus folder)
    CHANNEL_PATH - path to your channels file (usually called channelNames.txt). Only used for tif images with more than 3 channels, or 4D TIF images.
    NUCLEAR_CHANNEL_NAME - name of nuclear stain (corresponding to channelNames.txt).  Case sensitive.  Only used for tif images with more than 3 channels, or 4D TIF images.
    GROWTH_PIXELS - number of pixels from which to grow out from the nucleus to define a cell boundary.  Change based on tissue types.
    OUTPUT_METHOD - (imagej_text_file, statistics, visual_image_output, visual_overlay_output, all)
    BOOST - multiplier with which to boost the pixels of the nuclear stain before inference.  Choose 'auto' to try to infer the best boost to use based off of AUTOBOOST_PERCENTILE
    AUTOBOOST_REFERENCE_IMAGE - If autoboosting, then set this to the image's filename to choose which image to autoboost off of (generally choose a non-empty image).  If image not 
    found or empty, then just uses first filename to autoboost.  Does not set boost if BOOST is not set to 'auto', but gets metadata from selected image.
    MODEL_DIRECTORY - path to save logs to (not important)
    MODEL_PATH - path that contains your .h5 saved weights file for the model
    OVERLAP - amount of pixels overlap with which to run the stitching algorithm. Must be divisible by 2, and should be > cell diameter
    INCREASE_FACTOR - Amount with which to boost the size of the image. 2.5X works quite well after training on the Kaggle dataset.
    AUTOBOOST_PERCENTILE - The percentile value with which to saturate to.

    ---------OUTPUT PATHS-------------
    IMAGEJ_OUTPUT_PATH - path to output imagej .txt files
    QUANTIFICATION_OUTPUT_PATH - path to output .csv and .fcs quantifications
    VISUAL_OUTPUT_PATH - path to output visual masks as pngs.

    Note:  Unfortunately, ImageJ provides no way to export to the .roi file format needed to import into ImageJ.  Additionally, we can't
    use numpy in ImageJ scripts.  Thus, we need to write to file and read in (using the included imagej.py script) using the ImageJ
    scripter if we pick output to imagej_text_file
    '''
    # Change these!
    IS_CODEX_OUTPUT = True
    SHOULD_COMPENSATE = True
    target = './example_tile'
    output_path_name = "12_30_19"
    DIRECTORY_PATH = os.path.join(target, 'bestFocus')
    CHANNEL_PATH = os.path.join(target, 'channelNames.txt')
    NUCLEAR_CHANNEL_NAME = 'DNA1'
    GROWTH_PIXELS = 0
    OUTPUT_METHOD = 'all'
    BOOST = 'auto'
    AUTOBOOST_REFERENCE_IMAGE = 'reg001_X01_Y01_Z04.tif' #ie 'cellimage1.tif'

    # Usually not changed
    root = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIRECTORY = os.path.join(root, 'modelFiles')
    MODEL_PATH = os.path.join(root, 'src', 'modelFiles', 'final_weights.h5')
    IMAGEJ_OUTPUT_PATH = os.path.join(root, 'output', output_path_name, 'imagej_files')
    QUANTIFICATION_OUTPUT_PATH = os.path.join(root, 'output', output_path_name,'quantifications')
    VISUAL_OUTPUT_PATH = os.path.join(root, 'output', output_path_name,'visual_output')
    try:
        os.makedirs(IMAGEJ_OUTPUT_PATH)
        os.makedirs(QUANTIFICATION_OUTPUT_PATH)
        os.makedirs(VISUAL_OUTPUT_PATH)
    except FileExistsError:
        print("Directory already exists")
        
    OVERLAP = 80
    THRESHOLD = 20
    INCREASE_FACTOR = 2.5
    AUTOBOOST_PERCENTILE = 99.98

    # Probably don't change this, except the valid image extensions when working with unique extensions.
    def __init__(self):
        self.CHANNEL_NAMES = pd.read_csv(
            self.CHANNEL_PATH, sep='\t', header=None).values[:, 0]

        VALID_IMAGE_EXTENSIONS = ('tif', 'jpg', 'png')
        self.FILENAMES = [f for f in os.listdir(self.DIRECTORY_PATH) if f.endswith(
            VALID_IMAGE_EXTENSIONS) and not f.startswith('.')]
        if len(self.FILENAMES) < 1:
            raise NameError(
                'No image files found.  Make sure you are pointing to the right directory')
        
        reference_image_path = os.path.join(self.DIRECTORY_PATH, self.FILENAMES[0])

        if self.AUTOBOOST_REFERENCE_IMAGE != '' and self.BOOST == 'auto':
            if self.AUTOBOOST_REFERENCE_IMAGE in self.FILENAMES:
                self.FILENAMES.remove(self.AUTOBOOST_REFERENCE_IMAGE)
                self.FILENAMES.insert(0, self.AUTOBOOST_REFERENCE_IMAGE)
                print('Using autoboost reference image with filename', self.AUTOBOOST_REFERENCE_IMAGE)
            else:
                print('AUTOBOOST_REFERENCE_IMAGE does not exist.  Check your config file - image filename must match exactly.')
                print('Defaulting to first image reference...')

        self.N_DIMS, self.EXT, self.DTYPE, self.SHAPE, self.READ_METHOD = cvutils.meta_from_image(reference_image_path)
