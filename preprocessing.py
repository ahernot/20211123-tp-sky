import cv2
import os
import numpy as np
# import pandas as pd

DIRPATH = 'src/sky-images'
OUTPUT_DIR = 'output'
MASK_CLEAR = np.array([255, ]*3)



def to_csv ():
    files = os.listdir(DIRPATH)
    files_nb = len(files)
    files.sort()

    # Keep only images
    images = files[:int(files_nb/2)]

    pixels_sky = np.empty((0, 3))
    pixels_not = np.empty((0, 3))


    for image in images:

        image_name, _ = os.path.splitext(image)
        id = image_name[4:]
        mask = f'mask_{id}_skymask.png'

        image_path = os.path.join(DIRPATH, image)
        mask_path  = os.path.join(DIRPATH, mask)

        image_array = cv2.imread (image_path)
        mask_array  = cv2.imread (mask_path)
        
        image_array_flat = image_array.reshape(-1, 1, 3)# / 255.
        mask_array_flat  = mask_array.reshape(-1, 1, 3)# / 255.

        # Apply mask
        mask_sky = (mask_array_flat.squeeze() == MASK_CLEAR) [:, 0]
        mask_not = (mask_array_flat.squeeze() != MASK_CLEAR) [:, 0]

        image_masked_sky = image_array_flat.squeeze() [mask_sky, :]
        image_masked_not = image_array_flat.squeeze() [mask_not, :]


        pixels_sky = np.concatenate((pixels_sky, image_masked_sky))
        pixels_not = np.concatenate((pixels_not, image_masked_not))




    pixels_sky_labels = np.ones(pixels_sky.shape[0])
    pixels_sky_labeled = np.column_stack((pixels_sky, pixels_sky_labels))
    pixels_not_labels = np.zeros(pixels_not.shape[0])
    pixels_not_labeled = np.column_stack((pixels_not, pixels_not_labels))

    pixels_total_labeled = np.concatenate((pixels_sky_labeled, pixels_not_labeled), axis=0)

    # Export to csv
    output_path = os.path.join(OUTPUT_DIR, 'export.npy')
    np.save(output_path, pixels_total_labeled.astype(np.int))

# to_csv()
