from typing import Tuple

import cv2
import numpy as np
import spectral
import utm
import rasterio
import spectral.io.envi as envi

from geopy.distance import geodesic
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def main():
    dest_path = r'/home/kobi/PycharmProjects/SuperPointPretrainedNetwork/assets/raw_data/Julis_ortho_X160_Y856_x_655360_y_3510271.tif'
    dest_file = rasterio.open(dest_path)
    dest_image = dest_file.read().transpose(1, 2, 0)

    # cv2.imshow('a', dest_image[6500:7000, 6700:7000])
    # cv2.waitKey(-1)

    # cv2.imwrite('assets/data/target.png', dest_image[6500:7000, 6700:7000])
    cv2.imwrite('assets/data/target.png', dest_image[5000:8000, 5000:8000])

    source_envi_path = r'/home/kobi/PycharmProjects/SuperPointPretrainedNetwork/assets/raw_data/raw_60000_or'
    source_path = f'{source_envi_path}.hdr'
    source_file = spectral.io.envi.open(source_path)
    if source_file.nbands == 3:
        source_image = source_file.read_bands(np.arange(3))
    elif source_file.nbands == 270:
        rgb_wavelengths = ['1041.23', '1202.52', '1602.77']
        rgb_indices = [source_file.metadata['wavelength'].index(wave) for wave in rgb_wavelengths]
        source_image = source_file.read_bands(rgb_indices)
    else:
        raise ValueError(f'Unexpected number of input bands - {source_file.nbands}')

    norm_source_image = normalize_image(source_image)
    norm_resized_source_image = resize_by_src_dest_pixel_resolution_ration(norm_source_image, source_file, dest_file)

    # grayed_out_source_image = np.repeat(np.expand_dims(norm_source_image[..., 0], axis=-1),
    #                                     axis=2, repeats=3)

    # plt.imshow(norm_source_image)
    # plt.show()
    # plt.imshow(norm_resized_source_image)
    # plt.show()
    plt.imshow(norm_resized_source_image[..., 0], cmap='gray')
    plt.show()
    plt.imsave('assets/data/source.png', norm_resized_source_image)

    # print(dest_raster.xy(0,0))

    source_raster = rasterio.open(source_envi_path)


    # top_left = (31.69388, 34.66791)
    # bottom_right = (31.69313, 34.66818)
    top_left = (31.693, 34.669)
    bottom_right = (31.691, 34.667)

    topleft = utm.from_latlon(*top_left)[:2]
    bottomright = utm.from_latlon(*bottom_right)[:2]

    x1, y1 = dest_file.index(*topleft)
    x2, y2 = dest_file.index(*bottomright)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    plt.imshow(dest_image[x1: x2, y1: y2])
    plt.show()

    plt.imsave('assets/data/cropped_dest.png', dest_image[x1: x2, y1: y2])

    fig, ax = plt.subplots()
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.imshow(dest_image)
    ax.add_patch(rect)
    plt.savefig('assets/data/marked_cropped_dest.png')


def normalize_image(image: np.ndarray) -> np.ndarray:
    norm_source_image = (image - image.min()) / (image.max() - image.min())
    norm_source_image = (norm_source_image * 255).astype(np.uint8)

    return norm_source_image


def resize_by_src_dest_pixel_resolution_ration(image: np.ndarray, source_file: envi.BsqFile,
                                               dest_file: rasterio.DatasetReader) -> np.ndarray:
    pixel_resolution_x_src, pixel_resolution_y_src = calculate_pixel_resolution_from_envi(source_file)
    pixel_resolution_x_dest, pixel_resolution_y_dest = (abs(dest_file.meta['transform'].a * 100),
                                                        abs(dest_file.meta['transform'].e * 100))
    dest_to_src_resolution_ratio_x = pixel_resolution_x_dest / pixel_resolution_x_src
    dest_to_src_resolution_ratio_y = pixel_resolution_y_dest / pixel_resolution_y_src
    resized_shape = (int(image.shape[1] // dest_to_src_resolution_ratio_x),
                     int(image.shape[0] // dest_to_src_resolution_ratio_y))
    resized_source_image = cv2.resize(image, resized_shape, interpolation=cv2.INTER_LINEAR)

    return resized_source_image


def calculate_pixel_resolution_from_envi(file: envi.BsqFile) -> Tuple[float, float]:
    map_info = file.metadata['map info']
    latitude, longitude = float(map_info[4]), float(map_info[3])
    location = (latitude, longitude)

    shift_lat, shift_long = float(map_info[6]), float(map_info[5])

    step_x = (latitude, longitude + shift_long)
    step_y = (latitude + shift_lat, longitude)

    pixel_resolution_x = geodesic(location, step_x).meters * 100
    pixel_resolution_y = geodesic(location, step_y).meters * 100

    return round(pixel_resolution_x, 1), round(pixel_resolution_y, 1)


if __name__ == '__main__':
    main()
