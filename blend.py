from PIL import Image
import numpy
import blend_modes
import cv2
from tqdm import tqdm
from numba import jit
import os, fnmatch

v = '/Users/rohith/Downloads/VMD/'
o = '/Users/rohith/Downloads/results/train/'
out = '/Users/rohith/Downloads/Blend/'

@jit
def blend(files):
    for i in tqdm(files):
        # Import VMDed image
        background_img_float = cv2.imread(v+i,-1) #VMD Images
        background_img_float = cv2.cvtColor(background_img_float, cv2.COLOR_GRAY2RGBA).astype(float)
        
        # Import Original image
        foreground_img_float = cv2.imread(o+i,-1) #Original
        foreground_img_float = cv2.bitwise_not(foreground_img_float)
        foreground_img_float = cv2.cvtColor(foreground_img_float, cv2.COLOR_GRAY2RGBA).astype(float)
        
        # Blend images
        opacity = 0.7  # The opacity of the foreground that is blended onto the background is 70 %.
        blended_img_float = blend_modes.lighten_only(foreground_img_float,background_img_float, opacity)

        # Convert blended image back into PIL image
        blended_img = np.uint8(cv2.cvtColor(blended_img_float.astype(np.float32), cv2.COLOR_RGBA2GRAY))  # Image needs to be converted back to uint8 type for PIL handling.
        blended_img_raw = Image.fromarray(blended_img) 
        blended_img_raw.save(out+i)

 if __name__ == "__main__":
 	files = fnmatch.filter(os.listdir(v), '*.jpg')
    blend(files)