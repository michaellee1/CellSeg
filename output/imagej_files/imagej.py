from ij import IJ
from ij import ImagePlus
from ij.gui import Roi, PointRoi, PolygonRoi
from ij.plugin.frame import RoiManager

ROI_PIXEL_INPUT_FILE_PATH = "/media/raid/michael/CellVision/segmenter/CellVisionSegmenter/imagej_files/reg001_X05_Y02_Z05-coords.txt"

imp = IJ.getImage()
rm = RoiManager.getInstance()
if not rm:
	rm = RoiManager()

roi_raw_vals = []
with open(ROI_PIXEL_INPUT_FILE_PATH, "r") as f:
	input_content = f.read()
	roi_raw_vals = input_content.splitlines()

for roi_raw_val in roi_raw_vals:
	raw_roi = roi_raw_val.split(',')
	roi_x = [float(x) for x in raw_roi[0].split()]
	roi_y = [float(x) for x in raw_roi[1].split()]
	roi = PolygonRoi(roi_x, roi_y, Roi.FREEROI)
	rm.addRoi(roi)
