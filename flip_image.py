import numpy as np
from scipy import ndimage
from PIL import Image

name_in = 'figures/Right_center_lane_driving.jpg'
center_image = ndimage.imread(name_in)
center_image_mirror = np.fliplr(center_image)

name_out = 'figures/Mirror_Right_center_lane_driving.jpg'

im = Image.fromarray(center_image_mirror)
im.save(name_out)