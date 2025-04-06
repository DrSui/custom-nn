from PIL import Image
import numpy as np
import os

def ResizeIMG_64_64(img_path):
    img = Image.open(img_path)
    resized_img = img.resize((64,64))
    img_array = np.asarray(resized_img)
    return img_array

def ResizeIMG_64_64_Array(img_array):
    resized_img = img_array.resize((64,64))
    img_array = np.asarray(resized_img)
    return img_array
