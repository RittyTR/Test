# Group Members:
# Suhaila Kondappilly Aliyar,fdai7995,1492822
# Ritty Tharakkal Raphel,fdai7690,1459915 

#C:\Users\ritty\AppData\Local\Programs\Python\Python312
import sys
import numpy as np
import imageio.v3 as iio
from pathlib import Path

def convert_data(images_path, npz_file):
    images = list()
    labels = list()

    # Recursively search for images in the specified directory
    for file in Path(images_path).rglob('*.png'):
        name = file.name
        images.append(iio.imread(file))
        labels.append(int(name[0]))

    np.savez(npz_file, images=np.array(images), labels=np.array(labels))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 convert_data.py <path_to_images> <npz>")
        sys.exit(1)

    images_path = sys.argv[1]
    npz_file = sys.argv[2]

    convert_data(images_path, npz_file)
