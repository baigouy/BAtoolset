from functools import lru_cache
import copy

import matplotlib.pyplot as plt

from batoolset.tools.logger import TA_logger  # logging
logger = TA_logger()  # logging_level=TA_logger.DEBUG
from batoolset.files.tools import smart_name_parser
from batoolset.images.centralstore import ImageCentralStore
import random
import os
from builtins import super, int
import warnings
import skimage
from skimage import io
from PIL import Image
import tifffile  # open Zeiss .tif and .lsm files
import glob
from skimage.transform import rescale
from skimage.util import img_as_ubyte
import numpy as np
import json
from natsort import natsorted  # sort strings as humans would do
import xml.etree.ElementTree as ET  # to handle xml metadata of images
import base64
import io
import traceback
from skimage.morphology import white_tophat, black_tophat
from skimage.morphology import footprint_rectangle
import pathlib
import platform
import datetime as dt
import sys
from scipy import ndimage as ndi
import read_lif  # read Leica .lif files (requires numexpr)

# for future development
# np = None
# try:
#     np = __import__('cupy') # 3d accelerated numpy
# except:
#     np = __import__('numpy')

# see https://en.wikipedia.org/wiki/Feature_scaling
normalization_methods = ['Rescaling (min-max normalization)', 'Standardization (Z-score Normalization)',
                         'Mean normalization', 'Max normalization (auto)', 'Max normalization (x/255)',
                         'Max normalization (x/4095)', 'Max normalization (x/65535)',
                         'Rescaling based on defined lower and upper percentiles',
                         'None']
normalization_ranges = [[0, 1], [-1, 1]]


def guess_dimensions(img):
    if not isinstance(img, np.ndarray):
        return None

    if hasattr(img, 'metadata'):
        if 'dimensions' in img.metadata:
            if isinstance(img.metadata['dimensions'], str) and img.metadata['dimensions']:
                return img.metadata['dimensions']

    if len(img.shape) == 2:
        return 'hw'
    if len(img.shape) >= 4:  # here I really have no clue about the last ones

        # pb of that --> more than 9 dimensions will fail but will it ever occur
        dimensions = ''  # give a letter ??? because numbers will fail

        forbidden = 'dhwc'
        char = 'a'
        for d in range(len(img.shape)):
            # dimensions += str(d)
            if d!=0:
                char = _next_char(char, foridden_chars=forbidden)
                dimensions += char
            else:
                dimensions += char

        # it can be hwc or dhw at the end
        if img.shape[-1] < img.shape[-2] or img.shape[-1] < img.shape[-3]:
            end = 'hwc'
        else:
            end = 'dhw'

        dimensions = dimensions[:-3]
        dimensions += end

        return dimensions

        # scroll all but last
    if len(img.shape) == 3:
        # it can be hwc or dhw
        if img.shape[-1] < img.shape[-2] or img.shape[-1] < img.shape[-3]:
            return 'hwc'
        else:
            return 'dhw'

def _next_char(ch, foridden_chars='dhwc'):
    new_char = chr(ord(ch) + 1)
    if foridden_chars:
        if new_char in foridden_chars:
            while(new_char in foridden_chars):
                ch = new_char
                new_char= chr(ord(ch) + 1)
    return new_char



def get_saturated_pixels_mask(img, max=None):
    # good --> may need to fix pixel size auto or to take a frequency cut off in case of hot pixels -−> TODO
    if max is None:
        max = img.max()
    return img == max

def clip(img, tuple=None, min=None, max=None):
    '''
    Clips the values of an image within a specified range.

    Args:
        img (ndarray): Input image.
        tuple (tuple, optional): Tuple specifying the minimum and maximum values for clipping.
        min (scalar, optional): Minimum value for clipping.
        max (scalar, optional): Maximum value for clipping.

    Returns:
        ndarray: Clipped image.

    Examples:
        >>> image = np.array([[0.5, 1.2, 0.8],
        ...                   [1.9, 2.5, 1.0],
        ...                   [0.3, 0.7, 1.5]])
        >>> clipped_image = clip(image, tuple=(0.5, 1.5))
        >>> print(clipped_image)
        [[0.5 1.2 0.8]
         [1.5 1.5 1. ]
         [0.5 0.7 1.5]]
    '''

    if tuple is not None:
        # If tuple is provided, extract the minimum and maximum values from it.
        min = tuple[0]
        max = tuple[1]

    # Clip the image values within the specified range.
    img = np.clip(img, a_min=min, a_max=max)

    # Return the clipped image.
    return img


def invert(img):
    '''
    Inverts the values of an image.

    Args:
        img (ndarray): Input image.

    Returns:
        ndarray: Inverted image.

    Examples:
        >>> image = np.array([[0, 1, 0],
        ...                   [1, 0, 1],
        ...                   [0, 1, 0]], dtype=bool)
        >>> inverted_image = invert(image)
        >>> print(inverted_image)
        [[ True False  True]
         [False  True False]
         [ True False  True]]
    '''

    if not img.dtype == bool:
        # If the image is not boolean, compute the maximum and minimum values of the image,
        # take the negative of the image values, and shift the range so that the minimum value
        # becomes the maximum value and vice versa.
        mx = img.max()
        mn = img.min()
        img = np.negative(img) + mx + mn
    else:
        # If the image is boolean, invert its values using the bitwise NOT operator.
        img = ~img

    # Return the inverted image.
    return img

def clip_by_frequency(img, lower_cutoff=None, upper_cutoff=0.05, channel_mode=True):
    '''
    Clips the values of an image based on frequency cutoffs.

    Args:
        img (ndarray): Input image.
        lower_cutoff (float, optional): Lower frequency cutoff.
        upper_cutoff (float, optional): Upper frequency cutoff.
        channel_mode (bool, optional): Specifies whether to perform clipping on individual channels.

    Returns:
        ndarray: Clipped image.
    '''

    logger.debug(' inside clip ' + str(lower_cutoff) + str(upper_cutoff) + str(channel_mode))

    # Check if all cutoff values are 0 or None, in which case the image remains unchanged.
    # if lower_cutoff == upper_cutoff == 0 or lower_cutoff == upper_cutoff == None:
    #     logger.debug('clip: keep image unchanged')
    #     return img
    #
    # # Check if either lower_cutoff or upper_cutoff is None while the other is 0, in which case the image remains unchanged.
    # if (lower_cutoff is None and upper_cutoff == 0) or (upper_cutoff is None and lower_cutoff == 0):
    #     logger.debug('clip: keep image unchanged')
    #     return img

    if not lower_cutoff and not upper_cutoff:
        logger.debug('clip: keep image unchanged')
        return img

    logger.debug('chan mode ' + str(channel_mode))

    # If the maximum and minimum values of the image are the same, return the image as is.
    if img.max() == img.min():
        logger.debug('clip: keep image unchanged')
        return img

    # If channel_mode is True, perform clipping on individual channels.
    if channel_mode:
        for ch in range(img.shape[-1]):
            img[..., ch] = clip_by_frequency(img[..., ch], lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff,
                                                 channel_mode=False)
        return img

    logger.debug('Removing image outliers/hot pixels')

    # Compute the maximum value for clipping based on the upper_cutoff.
    if upper_cutoff is not None and upper_cutoff!=0:
        max = np.percentile(img, 100. * (1. - upper_cutoff))
        img[img > max] = max

    # Compute the minimum value for clipping based on the lower_cutoff.
    if lower_cutoff is not None and lower_cutoff!=0:
        min = np.percentile(img, 100. * lower_cutoff)
        img[img < min] = min

    return img


def get_2D_tiles_with_overlap(inp, width=512, height=512, overlap=0, overlap_x=None, overlap_y=None, dimension_h=0,
                              dimension_w=1, force_to_size=False):
    """
    Split a 2D and 3D input array into tiles with overlap along the specified dimensions.

    Args:
        inp (numpy.ndarray): The input 2D array.
        width (int): Width of each tile.
        height (int): Height of each tile.
        overlap (int): Overlap value to be applied along both x and y dimensions.
        overlap_x (int or None): Overlap value to be applied along the x dimension. If None, overlap value will be used.
        overlap_y (int or None): Overlap value to be applied along the y dimension. If None, overlap value will be used.
        dimension_h (int): Dimension index for the height (default: 0).
        dimension_w (int): Dimension index for the width (default: 1).
        force_to_size (bool): If True, forces the tiles to have the specified width and height.

    Returns:
        tuple: A tuple containing the crop parameters and the final split tiles.
            - crop_params (dict): Crop parameters including overlap values and dimensions.
            - final_splits (list): List of lists containing the split tiles.

    """

    if overlap_x is None:
        overlap_x = overlap
    if overlap_y is None:
        overlap_y = overlap

    if dimension_h < 0:
        dimension_h = len(inp.shape) + dimension_h
    if dimension_w < 0:
        dimension_w = len(inp.shape) + dimension_w

    final_height = inp.shape[dimension_h]
    final_width = inp.shape[dimension_w]

    if overlap_x % 2 != 0 or overlap_y % 2 != 0:
        logger.error('Warning overlap in x or y dimension is not even, this will cause numerous errors please do change this!')

    last_idx = 0
    cuts_y = []
    end = 0

    if height >= inp.shape[dimension_h]:
        overlap_y = 0
    if width >= inp.shape[dimension_w]:
        overlap_x = 0

    if height + overlap_y < inp.shape[dimension_h]:
        for idx in range(height, inp.shape[dimension_h], height):
            begin = last_idx
            end = idx + overlap_y
            if begin < 0:
                begin = 0
            if end >= inp.shape[dimension_h]:
                end = inp.shape[dimension_h]
            cuts_y.append((begin, end))
            last_idx = idx
        if end < inp.shape[dimension_h]:
            begin = last_idx
            end = inp.shape[dimension_h]
            if begin < 0:
                begin = 0
            cuts_y.append((begin, end))
    elif height + overlap_y > inp.shape[dimension_h]:
        height += overlap_y
        overlap_y = 0
        padding = []
        for dim in range(len(inp.shape)):
            padding.append((0, 0))
        padding[dimension_h] = (0, height - inp.shape[dimension_h])
        bigger = np.pad(inp, pad_width=tuple(padding), mode='symmetric')
        inp = bigger
        del bigger
        cuts_y.append((0, inp.shape[dimension_h]))
    else:
        # exactly one tile (height + overlap_y == image height): no overlap to trim at reassembly
        height += overlap_y
        overlap_y = 0
        cuts_y.append((0, inp.shape[dimension_h]))

    last_idx = 0
    cuts_x = []
    if width + overlap_x < inp.shape[dimension_w]:
        for idx in range(width, inp.shape[dimension_w], width):
            begin = last_idx
            end = idx + overlap_x
            if begin < 0:
                begin = 0
            if end >= inp.shape[dimension_w]:
                end = inp.shape[dimension_w]
            cuts_x.append((begin, end))
            last_idx = idx
        if end < inp.shape[dimension_w]:
            begin = last_idx
            end = inp.shape[dimension_w]
            if begin < 0:
                begin = 0
            cuts_x.append((begin, end))
    elif width + overlap_x > inp.shape[dimension_w]:
        width += overlap_x
        overlap_x = 0
        padding = []
        for dim in range(len(inp.shape)):
            padding.append((0, 0))
        padding[dimension_w] = (0, width - inp.shape[dimension_w])
        bigger = np.pad(inp, pad_width=tuple(padding), mode='symmetric')
        inp = bigger
        del bigger
        cuts_x.append((0, inp.shape[dimension_w]))
    else:
        # exactly one tile (width + overlap_x == image width): no overlap to trim at reassembly
        width += overlap_x
        overlap_x = 0
        cuts_x.append((0, inp.shape[dimension_w]))

    img_height = inp.shape[dimension_h]
    img_width = inp.shape[dimension_w]
    # plain 2D images (dimension_h == 0): the last row/column of tiles is shifted back so that it ends at the
    # image edge and is full size (real pixels instead of padding); shift_y/shift_x (how far back the last tile
    # starts) are stored in crop_params so that reassemble_tiles puts it back in place
    shift_last_tile = dimension_h == 0 and width + overlap_x <= img_width and height + overlap_y <= img_height
    shift_y = 0
    shift_x = 0
    if shift_last_tile:
        shift_y = cuts_y[-1][0] - (img_height - (height + overlap_y))
        shift_x = cuts_x[-1][0] - (img_width - (width + overlap_x))

    nb_tiles = 0
    final_splits = []
    for x_begin, x_end in cuts_x:
        cols = []
        for y_begin, y_end in cuts_y:
            slicer = [slice(None)] * len(inp.shape)
            if shift_last_tile and (y_end == img_height or x_end == img_width):
                slicer[dimension_h] = slice(y_end - (height + overlap_y), y_end)
                slicer[dimension_w] = slice(x_end - (width + overlap_x), x_end)
            else:
                slicer[dimension_h] = slice(y_begin, y_end)
                slicer[dimension_w] = slice(x_begin, x_end)
            cur_slice = inp[tuple(slicer)]
            nb_tiles += 1

            if not force_to_size:
                cols.append(cur_slice)
            else:
                padding = []
                for dim in range(len(cur_slice.shape)):
                    padding.append((0, 0))
                padding_required = False
                if cur_slice.shape[dimension_h] < height + overlap_y:
                    padding[dimension_h] = (0, (height + overlap_y) - cur_slice.shape[dimension_h])
                    padding_required = True
                if cur_slice.shape[dimension_w] < width + overlap_x:
                    padding[dimension_w] = (0, (width + overlap_x) - cur_slice.shape[dimension_w])
                    padding_required = True
                if padding_required:
                    bigger = np.pad(cur_slice, pad_width=tuple(padding), mode='symmetric')
                    cur_slice = bigger
                    del bigger

                cols.append(cur_slice)
        final_splits.append(cols)

    crop_params = {'overlap_y': overlap_y, 'overlap_x': overlap_x, 'final_height': final_height,
                   'final_width': final_width, 'n_cols': len(final_splits[0]), 'n_rows': len(final_splits),
                   'nb_tiles': nb_tiles, 'shift_y': shift_y, 'shift_x': shift_x}

    return crop_params, final_splits


def tiles_to_linear(tiles):
    """
    Convert a 2D array of tiles into a linear list.

    Args:
        tiles (list): A 2D list of tiles.

    Returns:
        list: A linear list of tiles.

    """
    linear = []
    for idx in range(len(tiles)):
        for j in range(len(tiles[0])):
            linear.append(tiles[idx][j])
    return linear


def tiles_to_batch(tiles):
    """
    Convert a 2D array of tiles into a batch.

    Args:
        tiles (list): A 2D list of tiles.

    Returns:
        ndarray: A batch array.

    """
    linear = tiles_to_linear(tiles)
    out = np.concatenate(tuple(linear), axis=0)
    return out



# https://en.wikipedia.org/wiki/Feature_scaling
def _normalize(img, individual_channels=False, method='Rescaling (min-max normalization)', norm_range=None,
              clip=False, normalization_minima_and_maxima=None):
    eps = 1e-20  # for numerical stability avoid division by 0

    if individual_channels:
        for c in range(img.shape[-1]):
            norm_min_max = None
            if normalization_minima_and_maxima is not None:
                if isinstance(normalization_minima_and_maxima[0], list):
                    norm_min_max = normalization_minima_and_maxima[c]
                else:
                    norm_min_max = normalization_minima_and_maxima
            img[..., c] = _normalize(img[..., c], individual_channels=False, method=method,
                                        norm_range=norm_range, clip=clip,
                                        normalization_minima_and_maxima=norm_min_max)
    else:
        if 'percentile' in method:
            if normalization_minima_and_maxima is None:
                lowest_percentile = np.percentile(img, norm_range[0])
                highest_percentile = np.percentile(img, norm_range[1])
            else:
                lowest_percentile = normalization_minima_and_maxima[0]
                highest_percentile = normalization_minima_and_maxima[1]

            try:
                import numexpr
                img = numexpr.evaluate("(img - lowest_percentile) / (highest_percentile - lowest_percentile + eps)")
            except Exception:
                img = (img - lowest_percentile) / (highest_percentile - lowest_percentile + eps)

            if clip:
                img = np.clip(img, 0, 1)
        elif method == 'Rescaling (min-max normalization)':
            max_val = img.max()
            min_val = img.min()

            if norm_range is None or norm_range == [0, 1] or norm_range == '[0, 1]' or norm_range == 'default' \
                    or isinstance(norm_range, int):
                try:
                    import numexpr
                    img = numexpr.evaluate("(img - min_val) / (max_val - min_val + eps)")
                except Exception:
                    img = (img - min_val) / (max_val - min_val + eps)
            elif norm_range == [-1, 1] or norm_range == '[-1, 1]':
                try:
                    import numexpr
                    img = numexpr.evaluate("-1 + ((img - min_val) * (1 - -1)) / (max_val - min_val + eps)")
                except Exception:
                    img = -1 + ((img - min_val) * (1 - -1)) / (max_val - min_val + eps)
        elif method == 'Mean normalization':
            max_val = img.max()
            min_val = img.min()

            if max_val != 0 and max_val != min_val:
                img = (img - np.average(img)) / (max_val - min_val)
        elif method.startswith('Max normalization'):
            if 'auto' in method:
                max_val = img.max()
            elif '255' in method:
                max_val = 255
            elif '4095' in method:
                max_val = 4095
            elif '65535' in method:
                max_val = 65535

            if max_val != 0:
                try:
                    import numexpr
                    img = numexpr.evaluate("img / max_val")
                except Exception:
                    img = img / max_val
        else:
            logger.error('Unknown normalization method "' + str(method) + '" --> ignoring ')

    return img


def _standardize(img, individual_channels=False, method=None, norm_range=range):
    if individual_channels:
        for c in range(img.shape[-1]):
            img[..., c] = _standardize(img[..., c], individual_channels=False, method=method,
                                           norm_range=norm_range)
    else:
        mean = np.mean(img)
        std = np.std(img)

        if std != 0.0:
            img = (img - mean) / std
        else:
            print('error empty image')
            if mean != 0.0:
                img = (img - mean)

    if norm_range == [0, 1] or norm_range == [-1, 1] or norm_range == '[0, 1]' or norm_range == '[-1, 1]':
        img = _normalize(img, method='Rescaling (min-max normalization)',
                            individual_channels=individual_channels, norm_range=[0, 1])

    if norm_range == [-1, 1] or norm_range == '[-1, 1]':
        img = (img - 0.5) * 2.

    logger.debug('max after standardization=' + str(img.max()) + ' min after standardization=' + str(img.min()))
    return img


# nb normalization assumes that the image is hwc --> and not hw --> maybe change that some day
def normalization(img, method=None, range=None, individual_channels=False, clip=False,
                  normalization_minima_and_maxima=None):
    """
    Normalize or standardize an image.

    Args:
        img (ndarray): The input image.
        method (str): The normalization method. Options: 'None' (no normalization),
            'percentile', 'normalization', 'standardization'.
        range (tuple): The range of values to normalize or standardize the image.
        individual_channels (bool): Flag indicating whether to normalize each channel
            individually. Default is False.
        clip (bool): Flag indicating whether to clip the values after normalization.
            Default is False.
        normalization_minima_and_maxima (list): List of tuples specifying the minimum
            and maximum values for normalization. Each tuple corresponds to a channel.
            Default is None.

    Returns:
        ndarray: The normalized or standardized image.

    """

    if img is None:
        logger.error("'None' image cannot be normalized")
        return

    logger.debug('max before normalization=' + str(img.max()) + ' min before normalization=' + str(img.min()))

    if method is None or method == 'None':
        logger.debug('Image is not normalized')
        return img

    if 'percentile' in method:
        logger.debug('Image will be normalized using percentiles')
        img = img.astype(np.float32)
        img = _normalize(img, individual_channels=individual_channels, method=method, norm_range=range, clip=clip,
                             normalization_minima_and_maxima=normalization_minima_and_maxima)
        logger.debug('max after normalization=' + str(img.max()) + ' min after normalization=' + str(img.min()))
        return img
    elif 'normalization' in method and not 'tandardization' in method:
        logger.debug('Image will be normalized')
        img = img.astype(np.float32)
        img = _normalize(img, individual_channels=individual_channels, method=method, norm_range=range)
        logger.debug('max after normalization=' + str(img.max()) + ' min after normalization=' + str(img.min()))
        return img
    elif 'tandardization' in method:
        logger.debug('Image will be standardized')
        img = img.astype(np.float32)
        img = _standardize(img, individual_channels=individual_channels, method=method, norm_range=range)
        logger.debug('max after standardization=' + str(img.max()) + ' min after standardization=' + str(img.min()))
        return img
    else:
        logger.error('Unknown normalization method: ' + str(method))
    return img


def reassemble_tiles(tiles, crop_parameters, three_d=False):
    '''
    Reassembles the tiles into a single image.

    Args:
        tiles (list): List of tiles.
        crop_parameters (dict): Dictionary containing crop parameters.
        three_d (bool): Flag indicating if the tiles are 3D.

    Returns:
        np.ndarray: Reassembled image.
    '''

    # Extract crop parameters
    overlap_y = crop_parameters['overlap_y']
    overlap_x = crop_parameters['overlap_x']
    final_height = crop_parameters['final_height']
    final_width = crop_parameters['final_width']
    # last tile shifted back to end at the image edge (plain 2D tiling), 0 otherwise
    shift_y = crop_parameters.get('shift_y', 0)
    shift_x = crop_parameters.get('shift_x', 0)

    cols = []
    for i in range(len(tiles)):
        cur_size = 0
        for j in range(len(tiles[0])):
            # Determine the y-slice for each tile
            if j == 0:
                if overlap_y != 0:
                    y_slice = slice(None, -int(overlap_y / 2))
                else:
                    y_slice = slice(None, None)
            elif j == len(tiles[0]) - 1:
                y_slice = slice(int(overlap_y / 2) + shift_y, None)
            else:
                if overlap_y != 0:
                    y_slice = slice(int(overlap_y / 2), -int(overlap_y / 2))
                else:
                    y_slice = slice(None, None)

            # Crop the tile along the y-axis
            if not three_d:
                tiles[i][j] = tiles[i][j][y_slice, ...]
                cur_size += tiles[i][j].shape[0]
            else:
                tiles[i][j] = tiles[i][j][:, y_slice, ...]
                cur_size += tiles[i][j].shape[1]

        # Stack the cropped tiles vertically
        if not three_d:
            cols.append(np.vstack(tuple(tiles[i])))
        else:
            cols.append(np.hstack(tuple(tiles[i])))

    cur_size = 0
    for i in range(len(cols)):
        if i == 0:
            if overlap_x != 0:
                x_slice = slice(None, -int(overlap_x / 2))
            else:
                x_slice = slice(None, None)
        elif i == len(cols) - 1:
            x_slice = slice(int(overlap_x / 2) + shift_x, None)
        else:
            if overlap_x != 0:
                x_slice = slice(int(overlap_x / 2), -int(overlap_x / 2))
            else:
                x_slice = slice(None, None)

        # Crop the stacked tiles along the x-axis
        if not three_d:
            if len(cols[i].shape) == 3:
                cols[i] = cols[i][:, x_slice]
            else:
                cols[i] = cols[i][:, x_slice, ...]
            cur_size += cols[i].shape[1]
        else:
            if len(cols[i].shape) == 3:
                cols[i] = cols[i][:, :, x_slice]
            else:
                cols[i] = cols[i][:, :, x_slice, ...]
            cur_size += cols[i].shape[2]

    # Create the final reassembled image
    if not three_d:
        return np.hstack(tuple(cols))[:final_height, :final_width]
    else:
        return np.dstack(tuple(cols))[:, :final_height, :final_width]


def linear_to_2D_tiles(tiles, crop_parameters):
    '''
    Converts a linear list of tiles into a 2D grid.

    Args:
        tiles (list): Linear list of tiles.
        crop_parameters (dict): Dictionary containing crop parameters.

    Returns:
        list: 2D grid of tiles.
    '''

    # Extract crop parameters
    n_rows = crop_parameters['n_rows']
    n_cols = crop_parameters['n_cols']
    # nb_tiles is unused, consider removing it

    output = []
    counter = 0
    for i in range(n_rows):
        try:
            cols = []
            for j in range(n_cols):
                # Append tile to the current column
                cols.append(tiles[counter])
                counter += 1
            # Append column to the output grid
            output.append(cols)
        except Exception:
            # Print traceback in case of an exception
            traceback.print_exc()
            pass

    return output

def img2Base64(img):
    """
    Convert an image or a pyplot image to a Base64-encoded string.

    Args:
        img (numpy.ndarray or None): The input image as a numpy array or None if the image is generated using pyplot.

    Returns:
        str: The Base64-encoded string representation of the image.

    """
    if img is not None:
        # Assume image
        buf = io.BytesIO()
        im = Image.fromarray(img)
        im.save(buf, format='png')
        buf.seek(0)  # Rewind file
        figdata_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return figdata_png
    else:
        # Assume pyplot image
        print('Please call this before plt.show() to avoid getting a blank output')
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')  # Remove unnecessary white space around graph
        buf.seek(0)  # Rewind file
        figdata_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return figdata_png

# below assumes channels last
def rgb_to_lab(rgb_image):
    """
    Convert an RGB image to LAB color space.

    Args:
        rgb_image (numpy.ndarray): RGB image as a numpy array.

    Returns:
        numpy.ndarray: LAB image as a numpy array.
    """

    # MEGA TODO -−> SHALL I FIX THE MAX FOR NORMALIZATION --> PROBABLY YES

    from skimage import color
    # Normalize the RGB image to [0, 1]
    tmp = rgb_image / rgb_image.max()

    # Convert the RGB image to LAB
    lab_image = color.rgb2lab(tmp)

    return lab_image

def rgb2gray(rgb):
    '''
    Converts an RGB image to grayscale.

    Args:
        rgb (ndarray): RGB image.

    Returns:
        ndarray: Grayscale image.

    Examples:
        # Example 1: Convert an RGB image to grayscale
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> gray_image = rgb2gray(rgb_image)
        >>> print(gray_image)
        [[ 76.2195 149.685   29.07  ]
         [126.9873 254.9745   0.    ]]

    '''

    # Create a dot product of the RGB channels with the corresponding conversion factors to obtain grayscale values.
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    return gray

def BGR_to_RGB(bgr):
    """
    Convert an image from BGR color space to RGB color space.

    Args:
        bgr (numpy.ndarray): The BGR image.

    Returns:
        numpy.ndarray: The RGB image.

    Examples:
        # Example 1: Convert a BGR image to RGB
        >>> bgr_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> rgb_image = BGR_to_RGB(bgr_image)
        >>> print(rgb_image)
        [[[  0   0 255]
          [  0 255   0]
          [255   0   0]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]
    """
    return bgr[..., ::-1]

def RGB_to_BGR(rgb):
    """
    Convert an image from RGB color space to BGR color space.

    Args:
        rgb (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The BGR image.

    Examples:
        # Example 1: Convert an RGB image to BGR
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> bgr_image = RGB_to_BGR(rgb_image)
        >>> print(bgr_image)
        [[[  0   0 255]
          [  0 255   0]
          [255   0   0]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]
    """
    return rgb[..., ::-1]

def RGB_to_GBR(rgb):
    """
    Convert an image from RGB color space to GBR color space.

    Args:
        rgb (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The GBR image.

    Examples:
        # Example 1: Convert an RGB image to GBR
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> gbr_image = RGB_to_GBR(rgb_image)
        >>> print(gbr_image)
        [[[  0 255   0]
          [  0   0 255]
          [255   0   0]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]
    """
    return rgb[..., [2, 0, 1]]

def RGB_to_GRB(rgb):
    """
    Convert an image from RGB color space to GRB color space.

    Args:
        rgb (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The GRB image.

    Examples:
        # Example 1: Convert an RGB image to GRB
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> grb_image = RGB_to_GRB(rgb_image)
        >>> print(grb_image)
        [[[  0 255   0]
          [255   0   0]
          [  0   0 255]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]

    """
    return rgb[..., [1, 0, 2]]

def RGB_to_RBG(rgb):
    """
    Convert an image from RGB color space to RBG color space.

    Args:
        rgb (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The RBG image.

    Examples:
        # Example 1: Convert an RGB image to RBG
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> rbg_image = RGB_to_RBG(rgb_image)
        >>> print(rbg_image)
        [[[255   0   0]
          [  0   0 255]
          [  0 255   0]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]
    """
    return rgb[..., [0, 2, 1]]

def RGB_to_BRG(rgb):
    """
    Convert an image from RGB color space to BRG color space.

    Args:
        rgb (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The BRG image.

    Examples:
        # Example 1: Convert an RGB image to BRG
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[127, 127, 127], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        >>> brg_image = RGB_to_BRG(rgb_image)
        >>> print(brg_image)
        [[[  0 255   0]
          [  0   0 255]
          [255   0   0]]
        <BLANKLINE>
         [[127 127 127]
          [255 255 255]
          [  0   0   0]]]


    """
    return rgb[..., [2, 0, 1]]

def interpolation_free_rotation(img, angle=90):
    """
    Perform rotation of an image without interpolation.

    Args:
        img (numpy.ndarray): The input image.
        angle (int or str): The rotation angle in degrees. If 'random', a random angle of 90, 180, or 270 degrees is chosen.

    Returns:
        numpy.ndarray: The rotated image.

    """
    if angle == 'random':
        angle = random.choice([90, 180, 270])
        return interpolation_free_rotation(img, angle=angle)
    else:
        if angle < 0:
            angle = 360 + angle

        if angle == 270:
            return np.rot90(img, 3)
        elif angle == 180:
            return np.rot90(img, 2)
        else:
            return np.rot90(img)

def get_voxel_conversion_factor(orig, return_111_if_none=True):
    """
    Computes the voxel size of an input image and returns it as a tuple of voxel dimensions.

    Parameters:
        orig (image): The input image.
        return_111_if_none (bool): A boolean specifying whether to return (1, 1, 1) if the voxel size cannot be retrieved.
                                   This parameter is optional and defaults to True.

    Returns:
        tuple: A tuple of voxel dimensions.
    """
    from batoolset.SQLites.tools import get_voxel_size
    from batoolset.ta.database.sql import table_exists_in_db

    # first try to load it from the db to gain time and not do this too often
    if isinstance(orig, str):
        potential_db = smart_name_parser(orig,'TA')
        # TODO --> do that for TA as well !!!! and pyTA -−> maybe create a voxel size table that contains the real voxel size instead of the ratios −−> THAT MAKES SENSE AND I CAN DO ALL THE MEASUREMENTS WITH THAT
        # if os.path.exists(os.path.join(potential_db,'pyTA.db')):
            # try recover the voxel conversion factor from the db rather than from the image
        if os.path.exists(os.path.join(potential_db, 'FISH.db')):
            # get it from the voxel size table
            db_path = os.path.join(potential_db, 'FISH.db')
            if table_exists_in_db(db_path, 'voxel_size'):
                # try read the values from the db
                # make sure to save in zyx order !!!
                voxel_size = get_voxel_size(db_path)
                if voxel_size is not None:
                    return voxel_size
        # elif  os.path.exists(os.path.join(potential_db, 'wing_segmenter.db'):

        # elif  os.path.exists(os.path.join(potential_db, 'ladybug_seg.db'):

    if isinstance(orig, str):
        orig = Img(orig)

    voxel_conversion_factor = None
    try:
        # Check if voxel size is None
        if orig.metadata['vz'] is None or orig.metadata['vy'] is None or orig.metadata['vx'] is None:
            voxel_conversion_factor = None
        # If voxel size is not None, store it in the voxel_conversion_factor variable
        else:
            voxel_conversion_factor = (orig.metadata['vz'], orig.metadata['vy'], orig.metadata['vx']) # this is how I should write voxel size to the stuff
            print('used voxel size for distance computation', voxel_conversion_factor)
    # If an exception is raised, print an error message
    except Exception:
        print('ERROR: voxel size could not be retrieved from file')
    # If voxel_conversion_factor is None and return_111_if_none is True, return (1, 1, 1)
    if return_111_if_none:
        if voxel_conversion_factor is None:
            print('voxel size not found --> assuming isotropic pixels')
            voxel_conversion_factor = np.asarray([1., 1., 1.])
    # Return the voxel_conversion_factor
    return voxel_conversion_factor

def _rotate_along_Z_axis(img):
    """
    Creates a set where the z and y axes are flipped, allowing for better detection of spots in the perpendicular direction.

    Parameters:
        img (array): The input image.

    Returns:
        set: A set with the flipped z and y axes.
    """
    return np.rot90(img, axes=(0,1))

def _recover_orig_after_rotation_along_z_axis(img):
    """
    Reverses the effect of _rotate_along_Z_axis to recover the original file.

    Parameters:
        img (array): The input image.

    Returns:
        array: The recovered original file.
    """
    return np.rot90(img, axes=(1,0))


def random_roll_images_to_avoid_spot_being_centered(images):
    if not isinstance(images, list):
        images = [images]
    img = images[0]
    axis = random.randrange(0, len(img.shape))
    random_roll_dist = random.randrange(0, img.shape[axis])
    for iii, img in enumerate(images):
        images[iii] = np.roll(img, random_roll_dist, axis=axis)
    return images


def generate_random_image(width=10, height=10, color_mode=None):
    """
    Generates a random black and white or RGB image of the specified width and height.

    Args:
        width (int): The preferred width of the image. The actual width may be smaller if the color_mode is 'bw'.
        height (int): The preferred height of the image.
        color_mode (str): The color mode of the image, either 'bw' for black and white or 'rgb' for RGB.

    Returns:
        numpy.ndarray: The generated image as a NumPy array.
    """
    # Determine the actual width of the image based on the color mode
    if color_mode is None:
        color_mode = random.choice(['bw','rgb'])

    # Generate a random image of the specified size and color mode
    if color_mode == 'bw':
        image = np.random.rand(height, width)
    else:
        image = np.random.rand(height, width, 3)

    # Scale the image values to the range [0, 255]
    image = (image * 255).astype(np.uint8)

    return image

def dilate_3D_spots_a_bit(single_channel_3D_mask,iterations=1):
    '''
    Dilates the blobs or 3D masks slightly. Avoid using more than 2 iterations to prevent the formation of large diamond artifacts.

    Parameters:
        single_channel_3D_mask (array): The input single-channel 3D mask.

    Returns:
        array: The dilated blobs or 3D masks.
    '''
    ball = ndi.generate_binary_structure(rank=3, connectivity=1)
    return ndi.binary_dilation(single_channel_3D_mask, ball, iterations=iterations)

def erode_3D_spots_a_bit(single_channel_3D_mask,iterations=1):
    """
    Erodes the 3D spots slightly in a single-channel 3D mask.

    Parameters:
        single_channel_3D_mask (array): The input single-channel 3D mask.
        iterations (int): The number of erosion iterations to apply. This parameter is optional and defaults to 1.

    Returns:
        array: The eroded 3D spots.
    """
    ball = ndi.generate_binary_structure(rank=3, connectivity=1)
    return ndi.binary_erosion(single_channel_3D_mask, ball, iterations=iterations)

# mode : ({nearest, wrap, reflect, mirror, constant})
# mode reflect is cool so that I don't have to zoom to increase the size  by default
# assumes image is hwc except if two channels then expect hw
# nb could be useful to deform only in the Z axis to do 3D data aug!!!!
def elastic_deform(image, displacement=None, axis=None, order=0, zoom=None, rotation=None, mode='reflect',
                   return_deformation_matrix=False):
    '''
    Applies elastic deformation to the input image.

    Args:
        image (ndarray): The input image to be deformed.
        displacement (ndarray): The displacement field to be used for deformation. If None, a random displacement field is generated.
        axis (tuple): The axes along which to apply the deformation. If None, defaults to (0, 1) for 2D images and (1, 2) for 3D images.
        order (int): The order of interpolation to be used for the deformation. Default is 0 (nearest-neighbor).
        zoom (float): The zoom factor to be used for the deformation. If None, no zooming is applied.
        rotation (float): The rotation angle to be used for the deformation. If None, no rotation is applied.
        mode (str): The boundary mode to be used for the deformation. Default is 'reflect'.
        return_deformation_matrix (bool): Whether or not to return the deformation matrix. Default is False.

    Returns:
        ndarray: The deformed image.
    '''
    import elasticdeform
    if axis is None:
        if len(image.shape) == 4:
            # assume 'dhwc'
            axis = (1, 2)
        else:
            # assume 'hw' or 'hwc'
            axis = (0, 1)
    if displacement is None:
        np.random.seed(random.randint(0,
                                      10000000))  # force numpy rand seed based on random seed so that if several instances are run and no parameter is passed the random numpy deformation would still be the same
        displacement = np.random.randn(2, 3, 3) * 25
    X_deformed = elasticdeform.deform_grid(image, displacement, axis=axis, order=order, mode=mode, rotate=rotation,
                                           zoom=zoom)
    if return_deformation_matrix:
        # maybe would be smart to return the complete parameter dict some day but ok for now
        return displacement, X_deformed  # displacement can be used to pass all the parameters to another image --> quite easy to do
    else:
        return X_deformed


def array_equal(array1, array2, success_message=None):
    """
    Check if two NumPy arrays are equal using the assert_array_equal function from the numpy.ma.testutils module.

    Args:
        array1 (numpy.ndarray): The first array to compare.
        array2 (numpy.ndarray): The second array to compare.
        success_message (str, optional): An optional success message to print if the arrays are equal.

    Raises:
        AssertionError: If the arrays are not equal.
    """
    # Import the assert_array_equal function from the numpy.ma.testutils module
    from numpy.ma.testutils import assert_array_equal

    # Call assert_array_equal to check if the arrays are equal
    assert_array_equal(array1, array2)

    # If a success message is provided, print it
    if success_message:
        print(success_message)


def array_not_equal(array1, array2, success_message=None):
    """
    Check if two NumPy arrays are not equal by asserting that they raise an AssertionError when compared using the
    assert_array_equal function from the numpy.ma.testutils module.

    Args:
        array1 (numpy.ndarray): The first array to compare.
        array2 (numpy.ndarray): The second array to compare.
        success_message (str, optional): An optional success message to print if the arrays are not equal.

    Raises:
        AssertionError: If the arrays are equal.
    """
    # Import the assert_array_equal and assert_raises functions from the numpy.ma.testutils and numpy.testing modules
    from numpy.ma.testutils import assert_array_equal
    from numpy.testing import assert_raises

    # Call assert_raises with AssertionError, assert_array_equal, and the two input arrays to check if they are not equal
    assert_raises(AssertionError, assert_array_equal, array1, array2)

    # If a success message is provided, print it
    if success_message:
        print(success_message)


def array_different(array1, array2, success_message=None):
    """
    Check if two NumPy arrays are different by calling the array_not_equal function with the same input arguments.

    Args:
        array1 (numpy.ndarray): The first array to compare.
        array2 (numpy.ndarray): The second array to compare.
        success_message (str, optional): An optional success message to print if the arrays are different.

    Raises:
        AssertionError: If the arrays are the same.
    """
    # Call the array_not_equal function with the same input arguments and return the result
    return array_not_equal(array1, array2, success_message=success_message)

def rois_equal(roi1, roi2):
    """Define what makes two ImagejRoi objects equal."""
    return (
        roi1.name == roi2.name
        and roi1.top == roi2.top
        and roi1.left == roi2.left
        and roi1.bottom == roi2.bottom
        and roi1.right == roi2.right
        and np.array_equal(roi1.integer_coordinates, roi2.integer_coordinates)
    )

def remove_duplicate_rois(rois):
    """Return a new list with duplicates removed."""
    unique = []
    for roi in rois:
        if not any(rois_equal(roi, u) for u in unique):
            unique.append(roi)
    return unique


def get_ImageJ_ROIs(filename, unpack_if_single_ROI=False, remove_dupes=True):
    """
    Load ImageJ ROI objects from a file using the roifile module.

    Args:
        filename (str): The name of the file containing the ROI objects.
        unpack_if_single_ROI (bool, optional): Whether to return a single ROI object instead of a list if the file
                                                contains only one ROI object. Defaults to False.

    Returns:
        List of ImagejRoi objects or a single ImagejRoi object if unpack_if_single_ROI is True and the file contains only
        one ROI object. Returns None if the file contains no ROI objects.
    """
    # Import the ImagejRoi class from the roifile module
    from roifile import ImagejRoi

    # Load the ROI objects from the file using the ImagejRoi.fromfile method
    rois = ImagejRoi.fromfile(filename)  # This line causes a bug but the ROIs seem OK # there seem to be a bug here the last ROI is now entered twice --> check

    if remove_dupes:
        rois = remove_duplicate_rois(rois)

    # If unpack_if_single_ROI is True and there is only one ROI object, return it directly instead of a list
    if unpack_if_single_ROI and rois is not None:
        if len(rois) == 0:  # If the file contains no ROI objects, return None
            return None
        if len(rois) == 1:  # If the file contains only one ROI object, return it directly
            return rois[0]

    # If unpack_if_single_ROI is False or there are multiple ROI objects, return the list of ROI objects
    return rois


def read_file_from_url(url):
    """
    Load a file from a URL (local file or remote) and return its contents as a bytes object or file name.

    Args:
        url (str): The URL of the file to load.

    Returns:
        If the URL starts with 'file:', returns the file name with the 'file:' prefix removed. Otherwise, returns a
        bytes object containing the contents of the file.
    """
    try:
        # If the URL is a string and starts with 'file:', assume it points to a local file and return the file name
        if isinstance(url, str):
            if url.lower().startswith('file:'):
                return url[7:]  # Trim 'file://' from the URL to get the file name

        # If the URL is not a local file, use the requests module to download its contents to a bytes object
        import requests
        resp = requests.get(url)
        bytes = io.BytesIO(resp.content)
        resp.close()
        return bytes

    # If an exception is raised during file loading, print the traceback and log an error message
    except Exception:
        import traceback
        traceback.print_exc()
        logger.error('could not load file from url ' + str(url))


# Opens an image using my custom tool for image annotation and display
def pop(img):
    """
    Open an image using a custom tool for image annotation and display.

    Args:
        img: The image to open and display.
    """
    # Set the UI to be used by the tool to qtpy
    from batoolset.settings.global_settings import set_UI
    set_UI()

    # Import necessary modules from qtpy and the custom tool
    from qtpy.QtWidgets import QApplication
    from batoolset.GUI.paint2 import Createpaintwidget
    from batoolset.GUI.scrollablepaint import scrollable_paint

    # Start a new QApplication instance
    app = QApplication(sys.argv)

    # Define a custom paint widget that overrides the default behavior of the tool's apply, shift_apply, ctrl_m_apply,
    # and save methods
    class overriding_apply(Createpaintwidget):
        def apply(self):
            pass

        def shift_apply(self):
            pass

        def ctrl_m_apply(self):
            pass

        def save(self):
            pass

    # Create a new scrollable_paint object with the custom paint widget, set the image to display, and freeze the panel
    # to prevent user changes
    w = scrollable_paint(custom_paint_panel=overriding_apply())
    w.set_image(img)
    w.freeze(True)

    # Show the image and start the application event loop
    w.show()
    app.exec_()


def avg_proj(image, axis=0):
    """
    Compute the average projection of an image along a specified axis.

    Args:
        image (numpy.ndarray): The image to compute the average projection of.
        axis (int, optional): The axis along which to compute the average projection. Defaults to 0.

    Returns:
        numpy.ndarray: The average projection of the image along the specified axis.
    """
    # Compute the average projection of the image along the specified axis using the np.mean() function
    return np.mean(image, axis=axis)


def max_proj(image, axis=0):
    """
    Compute the maximum projection of an image along a specified axis.

    Args:
        image (numpy.ndarray): The image to compute the maximum projection of.
        axis (int, optional): The axis along which to compute the maximum projection. Defaults to 0.

    Returns:
        numpy.ndarray: The maximum projection of the image along the specified axis.
    """
    # Compute the maximum projection of the image along the specified axis using the np.max() function
    return np.max(image, axis=axis)

def min_proj(image, axis=0):
    """
    Compute the minimum projection of an image along a specified axis.

    Args:
        image (numpy.ndarray): The image to compute the maximum projection of.
        axis (int, optional): The axis along which to compute the maximum projection. Defaults to 0.

    Returns:
        numpy.ndarray: The minimum projection of the image along the specified axis.
    """
    # Compute the maximum projection of the image along the specified axis using the np.max() function
    return np.min(image, axis=axis)


def is_binary(image):
    """
    Check if an image is binary (contains only two unique pixel values).

    Args:
        image: The image to check.

    Returns:
        bool: True if the image is binary, False otherwise.
    """
    # Find the maximum and minimum pixel values in the image
    mx = image.max()
    mn = image.min()

    # If the maximum and minimum values are equal, then the image is binary
    if mx == mn:
        return True

    # Count the number of pixels in the image that have the same value as the maximum or minimum pixel value
    binary_pixels = (image == mn) | (image == mx)
    num_binary_pixels = np.count_nonzero(binary_pixels)

    # If the number of binary pixels is equal to the total number of pixels in the image, then the image is binary
    if num_binary_pixels == image.size:
        return True

    # Otherwise, the image is not binary
    return False


def get_threshold_value_corresponding_to_percentile(cum_hist, bins, percentile_values_to_recover):
    """
    Compute the threshold value(s) corresponding to the specified percentile value(s) for a given cumulative histogram
    and bins.

    Args:
        cum_hist (numpy.ndarray): The cumulative histogram.
        bins (numpy.ndarray): The bins used to compute the histogram.
        percentile_values_to_recover (list or tuple): The percentile value(s) to recover the threshold value(s) for.

    Returns:
        The threshold value(s) corresponding to the specified percentile value(s).
    """
    out = []

    # If the percentile_values_to_recover argument is not a list or tuple, set it to an empty list
    if not isinstance(percentile_values_to_recover, (list, tuple)):
        percentile_values_to_recover = []

    # For each percentile value to recover, find the corresponding threshold value
    for val in percentile_values_to_recover:
        idx1 = np.argmax(
            cum_hist >= val)  # Find the index of the first element in the cumulative histogram that is greater than or equal to the percentile value
        out.append(bins[idx1])  # Append the corresponding bin value to the output list

    # If no threshold values were computed, return None
    if not out:
        return None

    # If only one threshold value was computed, return it
    if len(out) == 1:
        return out[0]

    # Otherwise, return the list of threshold values
    return out


def get_histogram_density_and_cum_histo_density(single_channel_img):
    """
    Compute the histogram, density, and cumulative density of a single-channel image.

    Args:
        single_channel_img (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the cumulative density, density, and bins of the image histogram.
    """
    #     # Compute the histogram and bins of the input image using the np.histogram() function
    #     hist, bins = np.histogram(single_channel_img, bins=np.arange(single_channel_img.min(), single_channel_img.max()),
    #                               density=True)  # density=True returns frequencies # old code not working for float imaghes between 0 and 1

    # Determine the number of bins (256 is a common choice, but you can adjust)
    num_bins = 256 if single_channel_img.dtype == np.float32 or single_channel_img.dtype == np.float64 else int(single_channel_img.max() - single_channel_img.min()) # fix for float images between 0 and 1, i.e. typically deep learning generated

    # Use linspace for float images, arange for integer images
    bins = np.linspace(single_channel_img.min(), single_channel_img.max(), num_bins + 1)

    # Compute the histogram with density=True to get the density
    hist, _ = np.histogram(single_channel_img, bins=bins, density=True)

    # Compute the cumulative histogram
    cum_hist = np.cumsum(hist) * np.diff(bins)  # Adjust for density scaling

    return cum_hist, hist, bins

# try an auto norm method à la ImageJ IJ FIJI --> somewhat the same idea as in https://github.com/imagej/ImageJ/blob/706f894269622a4be04053d1f7e1424094ecc735/ij/plugin/frame/ContrastAdjuster.java
def auto_scale(img, individual_channels=True, min_px_count_in_percent=0.005):
    """
    Auto-normalize the pixel intensities of an image based on their distribution.

    Args:
        img (numpy.ndarray): The input image.
        individual_channels (bool): Whether to normalize each color channel separately.
        min_px_count_in_percent (float): The minimum percentage of pixels to include in the normalization range.

    Returns:
        numpy.ndarray: The normalized image.
    """
    # If the input image is not a numpy array, return it unchanged
    if not isinstance(img, np.ndarray):
        return img

    # If the input image has multiple channels and individual_channels is True, normalize each channel separately
    if len(img.shape) > 2 and individual_channels:
        # Convert the parent image to float if it is not already
        if img.dtype != float:
            img = img.astype(float)
        for ch in range(img.shape[-1]):
            img[..., ch] = auto_scale(img[..., ch])
    else:
        # Convert the input image to float if it is not already
        if img.dtype != float:
            img = img.astype(float)
        mn = img.min()
        mx = img.max()
        if mn == mx:
            if mx != 0:
                return img / mx
            else:
                return img


        # print(mn, mx)

        # Compute the cumulative histogram, histogram, and bins of the input image using the get_histogram_density_and_cum_histo_density() function
        cum_hist, hist, bins = get_histogram_density_and_cum_histo_density(img)

        # print(cum_hist)
        # print(hist)
        # print(bins)

        # Compute the thresholds corresponding to the specified percentile values using the get_threshold_value_corresponding_to_percentile() function
        mn, mx = get_threshold_value_corresponding_to_percentile(cum_hist, bins,
                                                                 (min_px_count_in_percent, 1 - min_px_count_in_percent))
        # print(mn, mx) # --> bug here --> in fcat this can only work for values of type integer --> not cool


        # Normalize the image to the range between the computed thresholds and rescale to the original maximum value
        img = np.clip(img, mn, mx)
        # print('mn, mx',mn, mx)
        if mx-mn!=0:
            img = (img - mn) / (mx - mn)

    return img


def _create_dir(output_name):
    """
    Create a directory for the output file if it does not exist.

    Args:
        output_name (str): The path of the output file.

    Returns:
        None
    """
    # If no output name is provided, return immediately
    if output_name is None:
        return

    # Get the output folder and filename from the provided output name
    output_folder, filename = os.path.split(output_name)

    # If the output folder is not empty (i.e., if a parent folder is specified), create it if it does not exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)


# Be careful this modifies the original image --> maybe offer a copy
def fill_holes(img, fill_hole_below_this_size):
    """
    Fill holes in a binary image smaller than a specified size.

    Args:
        img (numpy.ndarray): The binary input image.
        fill_hole_below_this_size (int): The minimum size of holes to fill.

    Returns:
        numpy.ndarray: The binary image with holes filled.
    """
    from skimage.morphology import remove_small_holes
    # If the input image has multiple channels, fill holes in each channel separately
    if len(img.shape) == 3:
        # Assume HWC format and fill holes in all channels
        for ccc in range(img.shape[-1]):
            img[..., ccc] = fill_holes(img[..., ccc], fill_hole_below_this_size)
    else:
        # Create a binary mask from the input image
        mask = img > 0

        # Fill small holes in the mask using the remove_small_holes() function from scikit-image
        mask = remove_small_holes(mask, area_threshold=fill_hole_below_this_size, connectivity=1, out=mask)

        # Set the pixel values in the input image corresponding to the filled holes to the maximum pixel value
        img[mask != 0] = img.max()

    # Return the modified image with holes filled
    return img


# Be careful this modifies the original image --> maybe offer a copy
def clean_blobs_below(img, size_of_obejcts_to_be_removed):
    """
    Remove connected components in a binary image smaller than a specified size.

    Args:
        img (numpy.ndarray): The binary input image.
        size_of_obejcts_to_be_removed (int): The maximum size of objects to retain.

    Returns:
        numpy.ndarray: The binary image with small objects removed.
    """
    from skimage.morphology import remove_small_objects
    # If the input image has multiple channels, remove small objects in each channel separately
    if len(img.shape) == 3:
        # Assume HWC format and remove small objects in all channels
        for ccc in range(img.shape[-1]):
            img[..., ccc] = clean_blobs_below(img[..., ccc], size_of_obejcts_to_be_removed)
    else:
        # Create a binary mask from the input image
        mask = img > 0

        # Remove small objects from the mask using the remove_small_objects() function from scikit-image
        mask = remove_small_objects(mask, min_size=size_of_obejcts_to_be_removed, connectivity=1, in_place=False)

        # Set the pixel values in the input image corresponding to the removed small objects to the minimum pixel value
        img[mask == 0] = img.min()

    # Return the modified image with small objects removed
    return img


def to_stack_channel(*images):
    """
    Create a multi-channel image from single-channel ones.

    Args:
        *images (numpy.ndarray): The single-channel input images.

    Returns:
        numpy.ndarray: The multi-channel stacked image.
    """
    # Create a multi-channel image by stacking the input images along the last axis
    return np.stack(tuple(images), axis=-1)


def stack_to_imgs(arr):
    """
    Splits a 4-dimensional numpy array into a list of 3-dimensional numpy arrays.

    Args:
        arr (ndarray): A 4-dimensional numpy array of shape (n, h, w, c).

    Returns:
        A list of n 3-dimensional numpy arrays of shape (h, w, c).

    Examples:
        >>> array = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]], \
                     [[2, 2, 2], [2, 2, 2], [2, 2, 2]]])
        >>> print(stack_to_imgs(array))
        [array([[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]), array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]), array([[2, 2, 2],
               [2, 2, 2],
               [2, 2, 2]])]

    """
    # Split the 4-dimensional array into a list of 3-dimensional arrays
    arr_list = np.split(arr, arr.shape[0])

    # Convert the list of 3-dimensional arrays to a Python list
    arr_list = [arr.squeeze(axis=0) for arr in arr_list]

    return arr_list


def to_stack(images, channel=None, print_progress=False):
    """
    Create a stack of images along the first axis.

    Args:
        images (list): A list of input images or image paths.

    Returns:
        numpy.ndarray: The stacked image or None if input is empty.

    Examples:
        >>> image_paths = [np.zeros((3,3)), np.ones((3,3)), np.ones((3,3))+1]
        >>> print(image_paths[0].shape)
        (3, 3)
        >>> stack = to_stack(image_paths)
        >>> print(stack)
        [[[0. 0. 0.]
          [0. 0. 0.]
          [0. 0. 0.]]
        <BLANKLINE>
         [[1. 1. 1.]
          [1. 1. 1.]
          [1. 1. 1.]]
        <BLANKLINE>
         [[2. 2. 2.]
          [2. 2. 2.]
          [2. 2. 2.]]]
        >>> print(stack.shape)
        (3, 3, 3)

    """
    # If the input is empty, return None
    if images is None or not images:
        return

    # If the input is a list of image paths, convert them to images
    if isinstance(images[0], str):
        final_images = []
        for image in images:
            if print_progress:
                print('loading', image)
            tmp = Img(image)
            if channel is not None:
                tmp = tmp[...,channel]
            final_images.append(tmp)
            del tmp
        images = final_images
    else:
        if channel is not None:
            images = [image[..., channel] for image in images]

    if len(images) == 1:
        return images[0]

    # if channel is not None:
    #     images = [image[...,channel] for image in images]

    # Try to create a stack of images using np.stack()
    try:
        stack = np.stack(images, axis=0)
        return stack
    except Exception:
        # If the stack creation fails, print an error message and return None
        traceback.print_exc()
        print('Conversion to stack failed')


# NB this is memory safe but highly inefficient....
# def to_stack_from_list_low_mem(image_list, channel=None, print_progress=False):
#     """
#     Create a stack of images along the first axis.
#
#     Args:
#         images (list): A list of input images or image paths.
#
#     Returns:
#         numpy.ndarray: The stacked image or None if input is empty.
#
#     Examples:
#         >>> image_paths = [np.zeros((3,3)), np.ones((3,3)), np.ones((3,3))+1]
#         >>> print(image_paths[0].shape)
#         (3, 3)
#         >>> stack = to_stack(image_paths)
#         >>> print(stack)
#         [[[0. 0. 0.]
#           [0. 0. 0.]
#           [0. 0. 0.]]
#         <BLANKLINE>
#          [[1. 1. 1.]
#           [1. 1. 1.]
#           [1. 1. 1.]]
#         <BLANKLINE>
#          [[2. 2. 2.]
#           [2. 2. 2.]
#           [2. 2. 2.]]]
#         >>> print(stack.shape)
#         (3, 3, 3)
#
#     """
#     # If the input is empty, return None
#     if image_list is None or not image_list:
#         return
#
#     # If the input is a list of image paths, convert them to images
#     if isinstance(image_list[0], str):
#         # images = [Img(image_list[0])]
#         #
#         # if len(images) == 1:
#         #     return images[0]
#         #
#         # stacked = images[0][np.newaxis, ...]
#
#         stacked = None
#         for iii, img in enumerate(image_list):
#             if print_progress:
#                 print('loading',img)
#             tmp = Img(img)
#             if channel is not None:
#                 tmp = tmp[...,channel]
#             if stacked is None:
#                 stacked = tmp[np.newaxis, ...]
#             else:
#                 stacked = np.concatenate((stacked, tmp[np.newaxis, ...]), axis=0)
#         return stacked


def fake_n_channels(image, n_channels=3):
    """
    Create a n-channel image from a 2D or 3D grayscale image.

    Args:
        image (numpy.ndarray): The input image.
        n_channels (int, optional): The number of channels to create. Defaults to 3.

    Returns:
        numpy.ndarray: The n-channel image.
    """
    # If the input image is already an n-channel image, return it
    if len(image.shape) == 3 and image.shape[-1] == n_channels:
        return image

    # If the input image is a 2D grayscale image, create an n-channel image by copying the grayscale image to each channel
    elif len(image.shape) == 2:
        tmp = np.zeros((*image.shape, n_channels), dtype=image.dtype)
        for ch in range(n_channels):
            tmp[..., ch] = image
        return tmp

    # If the input image is neither a 2D grayscale image nor an n-channel image, return it unchanged
    else:
        return image


def has_metadata(im):
    '''
    checks if an image has metadata
    :param im: input image
    :return: True if an image is Img has metadata (false if Img was converted to a classical nd array that lost metadata)
    '''
    return hasattr(im, 'metadata')


def numpy_to_PIL(im, force_RGB=False):
    """
    Convert a numpy array to a PIL image.

    Args:
        im (numpy.ndarray): The input image as a numpy array.
        force_RGB (bool, optional): Whether to force the image to be in RGB format. Defaults to True.

    Returns:
        PIL.Image: The converted image as a PIL image.
    """
    # Create a PIL image from the input numpy array using the Image.fromarray() function
    img = Image.fromarray(im)

    # If force_RGB is True, convert the image to RGB format using the convert() method
    if force_RGB:
        img = img.convert('RGB')

    # Return the converted image as a PIL image
    return img


def PIL_to_numpy(PIL_image):
    """
    Convert a PIL image to a numpy array.

    Args:
        PIL_image (PIL.Image): The input image as a PIL image.

    Returns:
        numpy.ndarray: The converted image as a numpy array.
    """
    # Convert the PIL image to a numpy array using the np.array() function
    return np.array(PIL_image)


def fig_to_numpy(fig, tight=True):
    """
    Convert a matplotlib figure to a numpy array.

    Args:
        fig (matplotlib.figure.Figure): The input figure to convert.
        tight (bool, optional): Whether to use tight layout when saving the figure. Defaults to True.

    Returns:
        numpy.ndarray: The converted image as a numpy array.
    """
    # Create an in-memory buffer for saving the figure
    buf = io.BytesIO()

    # Save the figure to the buffer as a PNG image
    if tight:
        # Save with tight layout, useful for saving LUTs, for example
        fig.savefig(buf, format='png', bbox_inches='tight')
    else:
        fig.savefig(buf, format='png')

    # Reset the buffer position to the beginning
    buf.seek(0)

    # Open the buffer as an image and convert it to a numpy array
    im = np.array(Image.open(buf))

    # Close the buffer
    buf.close()

    # Return the converted image as a numpy array
    return im


def convolve(img, kernel=np.array([[-1, -1, -1],
                                   [-1, 8, -1],
                                   [-1, -1, -1]])):
    """
    Convolve an image using a given kernel.

    Args:
        img (numpy.ndarray): The input image as a numpy array.
        kernel (numpy.ndarray, optional): The convolution kernel to use. Defaults to a 3x3 Laplacian kernel.

    Returns:
        numpy.ndarray: The convolved image as a numpy array.
    """
    import scipy.signal     # here, not at module level: it costs ~1.5 s at every start
    # Use the convolve2d function from the scipy.signal module to convolve the image with the kernel
    convolved = scipy.signal.convolve2d(img, kernel, 'valid')

    # Return the convolved image as a numpy array
    return convolved


# TODO check on the different oses maybe replace mtime by mtime_nano
# https://docs.python.org/3/library/os.html#os.stat_result
def get_file_creation_time(filename, return_datetime_object=False):
    """
    Get the creation time of a file.

    Args:
        filename (str): The path or URL of the file to get the creation time for.
        return_datetime_object (bool, optional): Whether to return the creation time as a datetime object instead of a string. Defaults to False.

    Returns:
        str or datetime.datetime or None: The creation time of the file as a string or a datetime object, or None if the creation time could not be determined.
    """
    # Convert the filename to a pathlib.Path object
    fname = pathlib.Path(filename)

    # If the filename is a URL or starts with "file:", try to get the Last-Modified header from the URL
    if isinstance(filename, str):
        if filename.lower().startswith('http') or filename.lower().startswith('file:'):
            try:
                # Use the urlopen() function from the urllib.request module to open the URL and get the Last-Modified header
                from urllib.request import urlopen
                with urlopen(filename) as web_file:
                    return dict(web_file.getheaders())['Last-Modified']
            except Exception:
                # If getting the Last-Modified header fails, return None
                pass
            return None

    if not fname.exists() and isinstance(filename, str) and '#' in filename:
        # 'image.zarr#level=2': a fragment choosing what to read, not part of the name
        # ('#' is a valid file-name character, so only when the full name does not exist)
        fname = pathlib.Path(filename.split('#', 1)[0])

    # Check that the file exists using the exists() method of the pathlib.Path object
    assert fname.exists(), f'No such file: {fname}'

    # Get the modification time of the file using the st_mtime attribute of the stat() method of the pathlib.Path object
    mtime = dt.datetime.fromtimestamp(fname.stat().st_mtime)

    # If the platform is Windows, get the creation time of the file using the st_ctime attribute of the stat() method of the pathlib.Path object
    if 'indow' in platform.system():
        ctime = dt.datetime.fromtimestamp(fname.stat().st_ctime)
        if return_datetime_object:
            return ctime
        else:
            return str(ctime)

    # If the platform is not Windows, return the modification time as a datetime object or a string
    else:
        if return_datetime_object:
            return mtime
        else:
            return str(mtime)


def RGB_to_int24(RGBimg):
    '''
    Converts a 3-channel RGB image to a single-channel int24 image.

    Args:
        RGBimg (ndarray): A 3-channel RGB image of shape (height, width, 3).

    Returns:
        ndarray: A single-channel image of shape (height, width) with 24-bit integers.

    Examples:
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ...                       [[255, 255, 0], [255, 0, 255], [0, 255, 255]]], dtype=np.uint8)
        >>> int24_image = RGB_to_int24(rgb_image)
        >>> print(int24_image)
        [[16711680    65280      255]
         [16776960 16711935    65535]]
    '''
    RGB24 = (RGBimg[..., 0].astype(np.uint32) << 16) | (RGBimg[..., 1].astype(np.uint32) << 8) | RGBimg[..., 2].astype(
        np.uint32)
    return RGB24


def int24_to_RGB(RGB24):
    """
    Convert a 24-bit integer image to an RGB image.

    Args:
        RGB24 (numpy.ndarray): The input image as a numpy array of 24-bit integers.

    Returns:
        numpy.ndarray: The converted image as an RGB image with shape (h, w, 3).

    Examples:
        >>> int24_image = np.array([[16711680, 65280, 255],
        ...                         [16776960, 16711935, 65535]], dtype=np.uint32)
        >>> rgb_image = int24_to_RGB(int24_image)
        >>> print(rgb_image)
        [[[255   0   0]
          [  0 255   0]
          [  0   0 255]]
        <BLANKLINE>
         [[255 255   0]
          [255   0 255]
          [  0 255 255]]]
    """
    # Create an empty RGB image with the same shape as the input image
    RGBimg = np.zeros(shape=(*RGB24.shape, 3), dtype=np.uint8)

    # Convert each 24-bit integer value to R, G, and B components
    for c in range(RGBimg.shape[-1]):
        # Shift the bits of the 24-bit integer value by (2-c)*8 bits and mask with 0xFF to extract the component value
        RGBimg[..., c] = (RGB24 >> ((RGBimg.shape[-1] - c - 1) * 8)) & 0xFF

    # Return the converted image as an RGB image
    return RGBimg


# dirty code --> can i improve it ??
def _normalize_8bits(img, mode='min_max'):
    '''
    Normalizes the input image to 8-bit scale (0-255).

    Args:
        img (numpy.ndarray): The input image to be normalized.
        mode (str): The mode of normalization to be applied.
                    Possible values are 'min_max' and 'max'.
                    Default is 'min_max'.

    Returns:
        numpy.ndarray: The normalized image in 8-bit scale.
    '''
    try:
        if mode == 'min_max':
            if img.max() == img.min():
                return np.zeros_like(img, dtype=np.uint8)  # Return a black image if the input image is uniform
            img = (img - img.min()) / (
                        img.max() - img.min()) * 255.0  # Normalize the input image using min-max normalization
            img = img.astype(np.uint8)  # Convert the normalized image to 8-bit scale
        else:
            img /= img.max() / 255.0  # Normalize the input image using max normalization
            img = img.astype(np.uint8)  # Convert the normalized image to 8-bit scale
    except Exception:
        pass  # If there is an exception, just return the input image
    return img  # Return the normalized image as output


# can act as a nice data augmentation
def create_random_linear_gradient(img, pos_h=None, off_centered=True, mock=False):
    '''
    This function creates a random linear gradient and applies it to the input image.

    Args:
        img (numpy.ndarray): The input image to which the gradient will be applied.
        pos_h (float, optional): The horizontal position of the gradient. If not provided, the default is None.
        off_centered (bool, optional): A boolean that determines whether the gradient is off-centered or not. Default is True.
        mock (bool, optional): A boolean that determines whether to return the original image or to apply the augmentation.
                              If mock is True, the function returns the original image but calls random the same number of times
                              so that augmentation remains in sync.

    Returns:
        numpy.ndarray: The output of the function create_2D_linear_gradient(), which returns a gradient image.

    # Examples:
    #     >>> input_image = np.zeros((256, 256), dtype=np.uint8)
    #     >>> gradient_image = create_random_linear_gradient(input_image, pos_h=0.5, off_centered=True, mock=False)
    #     >>> print(gradient_image.shape)
    #     (256, 256)
    '''

    # Generate random values for the minimum and maximum values of the gradient, as well as the direction and position of the gradient.
    # The minimum and maximum values are uniformly distributed between 0.16 and 0.49 and between 0.5 and 1.0, respectively.
    # The direction is randomly selected to be either horizontal or vertical.
    # The position of the minimum value is randomly set to be at the top/left or bottom/right.
    return create_2D_linear_gradient(img, min=random.uniform(0.16, 0.49), max=random.uniform(0.5, 1.),
                                     horizontal=random.choice([True, False]),
                                     min_is_top_or_left=random.choice([True, False]), pos_h=pos_h,
                                     off_centered=off_centered, mock=mock)


# create a gradient image that can be applied to any 2D array
def create_2D_linear_gradient(img, min=0, max=1, horizontal=True, min_is_top_or_left=True, pos_h=None,
                              off_centered=False, mock=False):
    '''
       This function creates a 2D linear gradient on the input image.

       Args:
           img (numpy.ndarray): The input image.
           min (float, optional): The minimum value of the gradient. Default is 0.
           max (float, optional): The maximum value of the gradient. Default is 1.
           horizontal (bool, optional): If True, the gradient is horizontal; if False, the gradient is vertical. Default is True.
           min_is_top_or_left (bool, optional): If True, the minimum value is at the top or left; if False, the minimum value is at the bottom or right. Default is True.
           pos_h (int, optional): The position of the height dimension in the image shape.
           off_centered (bool, optional): If True, the gradient is shifted randomly along the gradient axis. Default is False.
           mock (bool, optional): If True, returns the original image but calls random the same number of times so that augmentation remains in sync. Default is False.

       Returns:
           numpy.ndarray: The gradient image.

       # Examples:
       #     >>> input_image = np.zeros((256, 256), dtype=np.uint8)
       #     >>> gradient_image = create_2D_linear_gradient(input_image, min=0.2, max=0.8, horizontal=True, min_is_top_or_left=True, off_centered=False)
       #     >>> print(gradient_image.shape)
       #     (256, 256)
    '''

    if img is None:
        return None
    if not min_is_top_or_left:
        min, max = max, min
    if pos_h is None:
        if len(img.shape) == 2:
            width = img.shape[1]
            height = img.shape[0]
        else:
            width = img.shape[-2]
            height = img.shape[-3]
    else:
        width = img.shape[pos_h + 1]
        height = img.shape[pos_h]
    g = None
    if not mock:
        if horizontal:
            # Create a horizontal gradient by tiling the linspace array along the height dimension.
            g = np.tile(np.linspace(min, max, width), (height, 1))
        else:
            # Create a vertical gradient by tiling the linspace array along the width dimension and transposing the result.
            g = np.tile(np.linspace(min, max, height), (width, 1)).T
    if off_centered:
        if horizontal:
            # Shift the gradient randomly along the horizontal axis.
            shift = random.randint(0, width)
            axis = 1
        else:
            # Shift the gradient randomly along the vertical axis.
            shift = random.randint(0, height)
            axis = 0
        if not mock:
            # Roll the gradient along the gradient axis.
            g = np.roll(g, shift, axis=axis)
    return g

def create_random_intensity_graded_perturbation(img, pos_h=None, off_centered=True, mock=False):
    '''
    This function creates a random intensity graded perturbation and applies it to the input image.

    Args:
        img (numpy.ndarray): The input image to which the perturbation will be applied.
        pos_h (int, optional): The horizontal position of the gradient. If not provided, the default is None.
        off_centered (bool, optional): A boolean that determines whether the gradient is off-centered or not. Default is True.
        mock (bool, optional): A boolean that determines whether to return the original image or to apply the augmentation. If mock is True, the function returns the original image but calls random the same number of times so that augmentation remains in sync.

    Returns:
        numpy.ndarray: The output of either create_random_gaussian_gradient() or create_random_linear_gradient(), which returns a random gradient image.

    '''

    # Randomly select between creating a linear gradient or a Gaussian gradient.
    function_to_run = random.choice([create_random_gaussian_gradient, create_random_linear_gradient])

    # Set the 'centered' boolean parameter based on whether the gradient is off-centered or not.
    centered = off_centered

    # If the selected function is create_random_liner_gradient(), set centered to False.
    if function_to_run == create_random_linear_gradient:
        centered = False

    # Apply the selected gradient function to the input image, using the provided or default parameters.
    return function_to_run(img, pos_h=pos_h, off_centered=centered, mock=mock)

def create_random_gaussian_gradient(img, pos_h=None, off_centered=True, mock=False):
    '''
    This function creates a random Gaussian gradient and applies it to the input image.

    Args:
        img (numpy.ndarray): The input image to which the gradient will be applied.
        pos_h (int, optional): The horizontal position of the gradient. If not provided, the default is None.
        off_centered (bool, optional): A boolean that determines whether the gradient is off-centered or not. Default is True.
        mock (bool, optional): A boolean that determines whether to return the original image or to apply the augmentation. If mock is True, the function returns the original image but calls random the same number of times so that augmentation remains in sync.

    Returns:
        numpy.ndarray: The output of the function gaussian_intensity_2D(), which creates a Gaussian gradient image.

    '''

    # If the input image is None, return None.
    if img is None:
        return None

    # Generate random values for the standard deviation and mean of the Gaussian distribution.
    # The standard deviation is uniformly distributed between 0.2 and 5.0, and the mean is uniformly distributed between -2.0 and 2.0.
    # These values are used as parameters for the gaussian_intensity_2D() function, which applies the Gaussian gradient to the input image.
    gaussian = gaussian_intensity_2D(img, sigma=random.uniform(0.2, 5.), mu=random.uniform(-2., 2.), pos_h=pos_h,
                                     off_centered=off_centered, mock=mock)

    # If mock is True, return the Gaussian gradient without any further processing.
    if mock:
        return gaussian

    # If the maximum value of the Gaussian gradient is less than 0.6, stretch the values to increase the signal strength.
    if gaussian.max() < 0.6:
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

    # If the range of values in the Gaussian gradient is less than 0.3, increase the range by renormalizing the values.
    if gaussian.max() - gaussian.min() < 0.3:
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

    # Clip the values in the Gaussian gradient to be within the range of 0.2 to 1.0.
    gaussian = np.clip(gaussian, 0.2, 1.)

    # Return the Gaussian gradient.
    return gaussian

# almost what I want --> but I have poor control on max an min values --> see how I can change that
# nb it is a gaussian only when width = height but as a data aug all of this is fine for me...
def gaussian_intensity_2D(img, sigma=1., mu=0., pos_h=None, off_centered=False, mock=False):
    '''
    This function applies a 2D Gaussian intensity gradient to the input image.

    Args:
        img (numpy.ndarray): The input image to which the gradient will be applied.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.0.
        mu (float, optional): The mean of the Gaussian distribution. Default is 0.0.
        pos_h (int, optional): The horizontal position of the gradient. If not provided, the default is None.
        off_centered (bool, optional): A boolean that determines whether the gradient is off-centered or not. Default is False.
        mock (bool, optional): A boolean that determines whether to return the original image or to apply the augmentation. If mock is True, the function returns None.

    Returns:
        numpy.ndarray: The 2D Gaussian intensity gradient.
    '''

    # If the input image is None, return None.
    if img is None:
        return None

    # Determine the width and height of the input image based on its shape and the value of pos_h.
    if pos_h is None:
        if len(img.shape) == 2:
            width = img.shape[1]
            height = img.shape[0]
        else:
            width = img.shape[-2]
            height = img.shape[-3]
    else:
        width = img.shape[pos_h + 1]
        height = img.shape[pos_h]

    # Initialize the Gaussian intensity gradient to None.
    g = None

    # If mock is False, create a 2D meshgrid that contains the x and y coordinates of each pixel in the image.
    # Compute the distance from the center of the meshgrid to each pixel and use it to compute the Gaussian intensity gradient.
    if not mock:
        x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    # If off_centered is True, randomly select either the x or y axis and shift the Gaussian intensity gradient by a random amount.
    if off_centered:
        axis = random.choice([0, 1])
        shift = random.randint(0, width if axis == 0 else height)
        if not mock:
            g = np.roll(g, shift, axis=axis)

    # Return the Gaussian intensity gradient.
    return g


# will that always work --> need think about it but seems to work with 2D images (single and multi channel) and with 3D images single channel --> ok for now maybe need be smarter if I add even more dimensions
def apply_2D_gradient(img, gradient2D):
    '''
        This function applies a 2D gradient to the input image.

        Args:
            img (numpy.ndarray): The input image to which the gradient will be applied.
            gradient2D (numpy.ndarray): The 2D gradient to apply to the input image.

        Returns:
            numpy.ndarray: The input image with the 2D gradient applied.

    '''

    # If the gradient is None, return the original image.
    if gradient2D is None:
        logger.debug('gradient is None, returning original')
        return img

    # If the image is None, return None.
    if img is None:
        logger.error('Image is None, gradient cannot be applied, sorry...')
        return None

    # Determine the dimensions of the input image and the gradient.
    nb_of_dims_img = len(img.shape)
    nb_of_dims_gradient = len(gradient2D.shape)

    dim = 0
    if nb_of_dims_img != nb_of_dims_gradient and nb_of_dims_gradient < nb_of_dims_img:
        # If the number of dimensions in the gradient is less than the number of dimensions in the image,
        # add dimensions to the gradient until they match.
        for dim in range(0, nb_of_dims_img):
            begin = nb_of_dims_img - dim - nb_of_dims_gradient
            end = nb_of_dims_img - dim
            if gradient2D.shape == img.shape[begin:end]:
                break

    # Add dimensions to the gradient until it has the same number of dimensions as the image.
    for dm in range(dim):
        gradient2D = gradient2D[..., np.newaxis]
    for dm in range(nb_of_dims_img - len(gradient2D.shape)):
        gradient2D = gradient2D[np.newaxis, ...]

    # Apply the gradient to the image.
    img[:, ...] = img[:, ...] * gradient2D

    # Return the image with the gradient applied.
    return img


# need recode all the save anyway and deduplicate code!!!
def save_as_tiff(img, output_name, print_file_name=False, ijmetadata='copy', mode='IJ', compression=None):
    '''
       Saves the current image as a TIFF file.

       Parameters:
           img (numpy.ndarray): The image to be saved.
           output_name (str): The name of the file to save.
           print_file_name (bool, optional): Whether to print the file name after saving. Default is False.
           ijmetadata (str, optional): The metadata type for ImageJ. Default is 'copy'.
           mode (str, optional): The saving mode. Default is 'IJ'.
           compression (str or None, optional): lossless TIFF compression passed to
               tifffile.imwrite (e.g. 'zlib', 'lzw'). Default None keeps images
               uncompressed (unchanged behaviour).

       # Examples:
       #     >>> image = np.zeros((256, 256), dtype=np.uint8)
       #     >>> save_as_tiff(image, "output_image.tif", print_file_name=True, ijmetadata='none')
       #     Saved file: output_image.tif
    '''

    if print_file_name:
        print('saving', output_name)

    if output_name is None:
        logger.error("No output name specified... ignoring...")
        return

    # TODO maybe handle tif with stars in their name here to avoid loss of data but ok for now...
    # if not '*' in output_name and (output_name.lower().endswith('.tif') or output_name.lower().endswith('.tiff')):
    _create_dir(output_name)
    if mode != 'IJ':  # TODO maybe do a TA mode or alike instead...
        out = img
        tifffile.imwrite(output_name, out, compression=compression)
    else:
        # create dir if does not exist
        out = img
        # apparently int type is not supported by IJ
        if out.dtype == np.int32:
            out = out.astype(np.float32)  # TODO check if correct with real image but should be
        if out.dtype == np.int64:
            out = out.astype(np.float64)  # TODO check if correct with real image but should be
        # IJ does not support bool type too
        if out.dtype == bool:
            out = out.astype(np.uint8) * 255
        if out.dtype == np.double:
            out = out.astype(np.float32)
        # if self.has_c():
        #     if not self.has_d() and self.has_t():
        #         out = np.expand_dims(out, axis=-1)
        #         out = np.moveaxis(out, -1, 1)
        #     out = np.moveaxis(out, -1, -3)
        #     tifffile.imwrite(output_name, out, imagej=True)  # make the data compatible with IJ
        # else:
        #     # most likely a big bug here --> fix it --> if has d and no t does it create a bug ???? --> maybe
        #     if not self.has_d() and self.has_t():
        #         out = np.expand_dims(out, axis=-1)
        #         out = np.moveaxis(out, -1, 1)
        #     out = np.expand_dims(out, axis=-1)
        #     # reorder dimensions in the IJ order
        #     out = np.moveaxis(out, -1, -3)
        #     tifffile.imwrite(output_name, out, imagej=True)  # this is the way to get the data compatible with IJ
        # should work better now and fix several issues... but need test it with real images
        # if image has no c --> assume all ok
        if getattr(img, '__dict__', None) is not None and 'metadata' in img.__dict__ and img.metadata[
            'dimensions'] is not None:
            # print('in dims')
            # print(self.has_c())  # why has no c channel ???
            if not img.has_c():
                out = out[..., np.newaxis]
            if not img.has_d():
                # Z goes AFTER time: a 'thw' movie must stay frames, not become slices
                out = np.expand_dims(out, axis=1 if img.has_t() else 0)
            if not img.has_t():
                out = out[np.newaxis, ...]
        else:
            # print('other')
            # no dimension specified --> assume always the same order that is tzyxc --> TODO maybe ...tzyxc
            if out.ndim < 3:
                out = out[..., np.newaxis]
            if out.ndim < 4:
                out = out[np.newaxis, ...]
            if out.ndim < 5:
                out = out[np.newaxis, ...]

        # print('final', out.shape)

        out = np.moveaxis(out, -1, -3)  # need move c channel before hw (because it is default IJ style)

        # TODO maybe offer compression at some point to gain space ???
        # imageJ order is TZCYXS order with dtype is uint8, uint16, or float32. Is S a LUT ???? probably yes because (S=3 or S=4) must be uint8. can I use compression with ImageJ's Bio-Formats import function.
        # TODO add the possibility to save ROIs if needed...
        #        Parameters 'append', 'byteorder', 'bigtiff', and 'imagej', are passed             #         to TiffWriter(). Other parameters are passed to TiffWriter.save().
        # print(ijmetadata)

        # working version 2021.11.2

        ijmeta = {}
        if getattr(img, '__dict__', None) is not None and 'metadata' in img.__dict__ and ijmetadata == 'copy':
            if img.metadata['Overlays']:
                ijmeta['Overlays'] = img.metadata['Overlays']
            if img.metadata['ROI']:
                ijmeta['ROI'] = img.metadata['ROI']
            # TODO add support for Luts some day --> make sure IJ luts and epyseg lust are not incompatible or define an IJ_LUTs in metadata and get it
            # make sure this does not create trouble
            if img.metadata['LUTs']:
                ijmeta['LUTs'] = img.metadata['LUTs']
        if not ijmeta:
            ijmeta = None

        # old save code with deprecated ijmetadata
        if tifffile.__version__ < '2022.4.22':
            tifffile.imwrite(output_name, out, imagej=True, ijmetadata=ijmeta,
                             compression=compression,
                             metadata={'mode': 'composite'} if getattr(img, '__dict__',
                                                                       None) is not None and 'metadata' in img.__dict__ and
                                                               img.metadata[
                                                                   'dimensions'] is not None and img.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
        else:
            try:
                # somehow this code doesn't seem to work with old tifffile but works with new one
                from tifffile.tifffile import imagej_metadata_tag
                # fix for ijmetadata deprecation in recent tifffile
                ijtags = imagej_metadata_tag(ijmeta, '>') if ijmeta is not None else {}
                # nb can add and save lut to the metadata --> see https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack

                # quick hack to force images to display as composite in IJ if they have channels -> probably needs be improved at some point
                tifffile.imwrite(output_name, out, imagej=True,
                                 metadata={'mode': 'composite'} if getattr(img, '__dict__',
                                                                           None) is not None and 'metadata' in img.__dict__ and
                                                                   img.metadata[
                                                                       'dimensions'] is not None and img.has_c() else {},
                                 compression=compression,
                                 extratags=ijtags)  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                # TODO at some point handle support for RGB 24-32 bits images saving as IJ compatible but skip for now
                # nb tifffile.imwrite(os.path.join(filename0_without_ext,'tra_test_saving_24bits_0.tif'), tracked_cells_t0, imagej=True,                      metadata={}) --> saves as RGB if image RGB 3 channels

                # TODO --> some day do the saving smartly with the dimensions included see https://pypi.org/project/tifffile/
                # imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',  compression = 'deflate', planarconfig = 'separate', tile = (32, 32),    metadata = {'axes': 'TZCYX'})
                # imwrite('temp.tif', volume, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),        metadata = {'spacing': 3.947368, 'unit': 'um', 'axes': 'ZYX'})
            except Exception:
                traceback.print_exc()
                tifffile.imwrite(output_name, out, imagej=True, compression=compression,
                                 metadata={'mode': 'composite'} if getattr(img, '__dict__',
                                                                           None) is not None and 'metadata' in img.__dict__ and
                                                                   img.metadata[
                                                                       'dimensions'] is not None and img.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3


# ImageJ metadata keys that are safe to carry over when rewriting a hyperstack
# (the counts images/channels/slices/frames and the ImageJ version are
# recomputed by tifffile itself, so they are intentionally not passed through)
_IJ_META_PASSTHROUGH = ('unit', 'spacing', 'finterval', 'fps', 'Info', 'Labels',
                        'Ranges', 'LUTs', 'mode', 'loop')


def recompress_tiff(path, compression='zlib', overwrite=True, skip_if_compressed=True,
                    force=False, predictor=False, quantize=None, print_info=True):
    '''Rewrite a TIFF with compression to reclaim disk space.

    By default the pixel data is preserved exactly (verified byte-for-byte
    before the original is replaced) and the ImageJ hyperstack structure /
    metadata (dimension order, unit, spacing, display Ranges, LUTs, Info,
    composite mode...) is carried over when present, as is the pixel resolution.

    The new file is written to a temporary file in the same folder and only
    swapped over the original once it is confirmed smaller and (for the lossless
    modes) pixel-identical, so an interrupted run can never corrupt or truncate
    the source file.

    Parameters
    ----------
    path : str
        Path of the .tif/.tiff file to recompress.
    compression : str
        Lossless tifffile codec, e.g. 'zlib' (default) or 'lzw'.
    overwrite : bool
        If True replace the original in place. If False write the compressed
        copy next to it (as '<name>_<compression>.tif', or '<name>_q<bits>.tif'
        when quantizing) and leave the original.
    skip_if_compressed : bool
        Skip files that already use a non-raw compression (unless force=True).
    force : bool
        Rewrite even if the result is not smaller / already compressed.
    predictor : bool
        Enable the TIFF predictor (horizontal for integers, floating-point for
        floats) before compression. Fully lossless. Helps continuous-tone /
        float data (e.g. ~1.2x -> ~1.4x on prediction maps) but is neutral or
        slightly counter-productive on sparse binary masks, so it defaults to
        False. If the predicted rewrite ends up larger it is discarded and a
        plain (predictor-less) rewrite is used instead.
    quantize : None, 8 or 16
        Defaults to None: no quantization is ever performed and the fully
        lossless path is always used. Must be set explicitly, per call, to opt in.
        LOSSY. When 8 or 16 and the image is floating-point, the data is linearly
        scaled from its [min, max] range to uint8 / uint16 before compression.
        This is meant for deep-learning float outputs (e.g. probability maps),
        which compress poorly as raw float32 (~1.2x) but shrink dramatically once
        quantized (typically >10x at 8-bit). The value range and original dtype
        are recorded in the file (ImageJ 'Info' tag, or the TIFF description) so
        the approximate floats can be reconstructed:
            float ~= vmin + code / (2**bits - 1) * (vmax - vmin)
        Quantization is ignored (lossless path used) for non-float images.
        Because it is lossy, verification is tolerance-based (max error <= half a
        quantization step) rather than byte-exact.
    print_info : bool
        Print a one-line before/after summary.

    Returns
    -------
    dict with keys: path, done (bool), skipped (bool), reason (str|None),
    old_size, new_size (bytes; new_size is None when skipped). When quantize is
    used it additionally carries: lossy (True), max_abs_error and quant_step
    (in the original value units).
    '''
    result = {'path': path, 'done': False, 'skipped': True, 'reason': None,
              'old_size': None, 'new_size': None}

    if path is None or not (path.lower().endswith('.tif') or path.lower().endswith('.tiff')):
        result['reason'] = 'not a .tif/.tiff file'
        if print_info:
            logger.error('recompress_tiff: %s', result['reason'])
        return result
    if not os.path.isfile(path):
        result['reason'] = 'file not found'
        if print_info:
            logger.error('recompress_tiff: file not found %s', path)
        return result

    old_size = os.path.getsize(path)
    result['old_size'] = old_size

    try:
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            data = series.asarray()
            axes = series.axes
            ij = tf.imagej_metadata
            page = tf.pages[0]
            photometric = page.photometric
            already_compressed = int(page.compression) != 1  # 1 == COMPRESSION.NONE

            def _rational(tagname):
                tag = page.tags.get(tagname)
                if tag is None:
                    return None
                v = tag.value
                try:
                    return v[0] / v[1]
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return None

            xres, yres = _rational('XResolution'), _rational('YResolution')
            resunit_tag = page.tags.get('ResolutionUnit')
            resunit = resunit_tag.value if resunit_tag is not None else None
    except Exception:
        traceback.print_exc()
        result['reason'] = 'could not read file'
        return result

    if already_compressed and skip_if_compressed and not force:
        result['reason'] = 'already compressed'
        if print_info:
            print('recompress_tiff: skipping already-compressed %s (%.1f KB)' % (path, old_size / 1024))
        return result

    resolution = (xres, yres) if (xres and yres) else None

    # LOSSY quantization only makes sense for floating-point images
    do_quant = quantize in (8, 16) and np.issubdtype(data.dtype, np.floating)
    if quantize in (8, 16) and not do_quant and print_info:
        print('recompress_tiff: quantize ignored for non-float image %s' % path)

    # build the array to write (quantized codes, or the untouched data)
    if do_quant:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        levels = (1 << quantize) - 1
        qdtype = np.uint8 if quantize == 8 else np.uint16
        if vmax > vmin:
            scaled = (np.nan_to_num(data, nan=vmin) - vmin) / (vmax - vmin)
            write_data = np.clip(np.rint(scaled * levels), 0, levels).astype(qdtype)
        else:
            write_data = np.zeros(data.shape, dtype=qdtype)
        quant_marker = '[batoolset_quantized] bits=%d vmin=%r vmax=%r orig_dtype=%s' % (
            quantize, vmin, vmax, str(data.dtype))
    else:
        write_data = data
        quant_marker = None

    # write to a temp file in the same directory so os.replace is atomic
    tmp = path + '.recompress_tmp.tif'
    try:
        if ij is not None:
            meta = {'axes': axes}
            # value-domain keys (Ranges/LUTs) no longer apply after quantization
            passthrough = _IJ_META_PASSTHROUGH if not do_quant else (
                'unit', 'spacing', 'finterval', 'fps', 'loop')
            for k in passthrough:
                if k in ij:
                    meta[k] = ij[k]
            if quant_marker is not None:
                existing = ij.get('Info') if ij else None
                meta['Info'] = (existing + '\n' + quant_marker) if existing else quant_marker
            kwargs = dict(imagej=True, metadata=meta, compression=compression)
            if resolution:
                kwargs['resolution'] = resolution
        else:
            kwargs = dict(compression=compression, photometric=photometric)
            if quant_marker is not None:
                kwargs['description'] = quant_marker
            if resolution:
                kwargs['resolution'] = resolution
                if resunit is not None:
                    kwargs['resolutionunit'] = resunit
        if predictor:
            kwargs['predictor'] = True
        tifffile.imwrite(tmp, write_data, **kwargs)
        # the predictor can occasionally enlarge a file (e.g. sparse masks) --
        # if so, fall back to a plain rewrite so we never lose the size gain
        if predictor and os.path.getsize(tmp) >= old_size:
            kwargs.pop('predictor', None)
            tifffile.imwrite(tmp, write_data, **kwargs)

        # verify before touching the original: byte-exact when lossless, or
        # within half a quantization step when quantizing
        with tifffile.TiffFile(tmp) as tf2:
            check = tf2.series[0].asarray()
        if check.shape != write_data.shape:
            raise ValueError('verification failed: shape changed on rewrite')
        if do_quant:
            if vmax > vmin:
                recon = vmin + check.astype(np.float32) / levels * (vmax - vmin)
            else:
                recon = np.full(data.shape, vmin, dtype=np.float32)
            max_err = float(np.nanmax(np.abs(recon - data))) if data.size else 0.0
            step = (vmax - vmin) / levels if vmax > vmin else 0.0
            if max_err > step * 0.5 + 1e-6 * (abs(vmax) + abs(vmin) + 1.0):
                raise ValueError('quantization error larger than expected (%.3g > %.3g)' % (max_err, step * 0.5))
            result['lossy'] = True
            result['max_abs_error'] = max_err
            result['quant_step'] = step
        elif not np.array_equal(check, data):
            raise ValueError('verification failed: recompressed pixels differ from source')

        new_size = os.path.getsize(tmp)
        if new_size >= old_size and not force:
            os.remove(tmp)
            result['reason'] = 'no size gain'
            if print_info:
                print('recompress_tiff: no gain for %s (%.1f KB), left unchanged' % (path, old_size / 1024))
            return result

        if overwrite:
            os.replace(tmp, path)  # atomic on same filesystem
            final_path = path
        else:
            base = path[:-5] if path.lower().endswith('.tiff') else path[:-4]
            suffix = ('_q%d' % quantize) if do_quant else ('_' + str(compression))
            final_path = base + suffix + '.tif'
            os.replace(tmp, final_path)

        result.update(done=True, skipped=False, path=final_path, new_size=new_size)
        if print_info:
            if do_quant:
                print('recompress_tiff: %s  %.1f KB -> %.1f KB (%.1fx)  LOSSY q%d, max_abs_error=%.4g' % (
                    final_path, old_size / 1024, new_size / 1024, old_size / max(new_size, 1),
                    quantize, result['max_abs_error']))
            else:
                print('recompress_tiff: %s  %.1f KB -> %.1f KB (%.1fx)' % (
                    final_path, old_size / 1024, new_size / 1024, old_size / max(new_size, 1)))
        return result
    except Exception:
        traceback.print_exc()
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        result['reason'] = 'error during recompression (original left untouched)'
        return result


def recompress_tiffs(paths, compression='zlib', recursive=True, overwrite=True,
                     skip_if_compressed=True, force=False, predictor=False,
                     quantize=None, print_info=True):
    '''Batch-recompress many TIFFs. Accepts a folder, a single file, or a list
    of paths/folders. Returns a summary dict with total bytes saved.

    Folders are scanned for *.tif/*.tiff (recursively when recursive=True).
    Each file is processed with recompress_tiff (see it for the guarantees,
    the lossless `predictor` option and the lossy `quantize` option).
    '''
    if isinstance(paths, str):
        paths = [paths]

    files = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _dirs, names in os.walk(p):
                    for n in names:
                        if n.lower().endswith(('.tif', '.tiff')):
                            files.append(os.path.join(root, n))
            else:
                for n in os.listdir(p):
                    fp = os.path.join(p, n)
                    if os.path.isfile(fp) and n.lower().endswith(('.tif', '.tiff')):
                        files.append(fp)
        elif os.path.isfile(p):
            files.append(p)

    summary = {'n_total': len(files), 'n_done': 0, 'n_skipped': 0,
               'bytes_before': 0, 'bytes_after': 0, 'results': []}
    for fp in files:
        r = recompress_tiff(fp, compression=compression, overwrite=overwrite,
                            skip_if_compressed=skip_if_compressed, force=force,
                            predictor=predictor, quantize=quantize, print_info=print_info)
        summary['results'].append(r)
        if r['done']:
            summary['n_done'] += 1
            summary['bytes_before'] += r['old_size']
            summary['bytes_after'] += r['new_size']
        else:
            summary['n_skipped'] += 1

    if print_info:
        saved = summary['bytes_before'] - summary['bytes_after']
        print('recompress_tiffs: %d/%d recompressed, %d skipped, saved %.1f MB' % (
            summary['n_done'], summary['n_total'], summary['n_skipped'], saved / (1024 * 1024)))
    return summary


# first attempt to make a one hot encoder --> probably needs some love though (especially with the values maybe getting the histogram of the image would be smarter than what I do)
# do it smartly so that it can even combine layers using additions for example --> have I done image math ???
# or do it after because simpler
# in a way it's just some channel swapping --> easy to do
def one_hot_encoder(img, remap_dict=None):
    '''
    Converts the input image to one-hot-encoded format.

    Args:
        img (numpy.ndarray): The input image to be one-hot-encoded.
        remap_dict (dict): A dictionary that maps input channel values to output channel values.
                           Default is None.

    Returns:
        numpy.ndarray: The one-hot-encoded image.
    '''
    # remap_dict = {2:1, 1:2, 0:0}  # Example of remap_dict dictionary

    one_hot_encoded = np.empty((*img.shape, img.max()),
                               dtype=np.uint8)  # Create an empty array to hold the one-hot-encoded image
    for ch, iii in enumerate(range(img.min(), img.max() + 1)):
        # Loop over each input channel value and create an output channel for it
        if remap_dict is not None:
            final_ch = remap_dict[
                ch]  # Map the input channel value to the corresponding output channel value using the remap_dict dictionary
        else:
            final_ch = ch  # If no remap_dict is provided, use the input channel value as the output channel value

        one_hot_encoded[
            img == iii, final_ch] = 255  # Set the pixels with the input channel value to 255 in the corresponding output channel

    return one_hot_encoded  # Return the one-hot-encoded image


# NB I could make a smarter version that takes just the channel of interest and the specific position into account ???? think about that
# NB I ASSUME IMAGE IS hw or hwc and nothing else which may be wrong!!! but then conversion should take place before the image is passed there
# z_behaviour='middle'
# TODO clean this code and remove useless parts
def toQimage(img, autofix_always_display2D=True, normalize=True, z_behaviour=None, metadata=None, preserve_alpha=False, dimensions=None, luts=None):
    '''
       Converts a numpy ndarray to a QImage.

       Parameters:
           img (numpy.ndarray): The input image as a numpy ndarray.
           autofix_always_display2D (bool, optional): Whether to automatically adjust the image for 2D display. Default is True.
           normalize (bool, optional): Whether to normalize the image values. Default is True.
           z_behaviour (None or str, optional): The z-behaviour for 3D images. Default is None.
           metadata (None or dict, optional): Metadata associated with the image. Default is None.
           preserve_alpha (bool, optional): Whether to preserve the alpha channel if present. Default is False.

       Returns:
           qimage: A PyQt compatible image (QImage).

    '''

    from batoolset.settings.global_settings import set_UI  # set the UI to qtpy
    set_UI()
    from qtpy.QtGui import QImage  # moved import here to make the class independent of pyqt
    # qimage = None

    # luts = luts
    if luts == None:
        try:
            luts = img.metadata['LUTs']
        except Exception:
            pass
        if metadata is not None:
            try:
                luts = metadata['LUTs']
            except Exception:
                pass

    logger.debug('Creating a qimage from a numpy image')
    # dimensions = dimensions
    if not dimensions:
        if isinstance(img, Img):
            try:
                dimensions = img.get_dimensions_as_string()
            except Exception:
                # fall back to hw dimensions if unknown as it is the default TA format anyawqy --> there are probably better solutions and I should not let images be built without dimensions but ok for now
                # dimensions = 'hw'
                if len(img.shape) == 2:
                    dimensions = 'hw'
                elif len(img.shape) == 3:
                    dimensions = 'hwc'
                else:
                    # could even go further --> assume up to tdhwc
                    logger.error('unknown image dimensions')
                    return

    img = np.copy(img)  # need copy the array

    if autofix_always_display2D and dimensions is not None:
        if 'h' in dimensions and dimensions.index('h') != 0:
            for dim in dimensions:
                if dim == 'h':
                    break
                if ((dim != 'd' and dim != 'z') or z_behaviour != 'middle'):
                    img = img[0]  # always take the first image execpt for Z stack
                else:
                    img = img[int(img.shape[0] / 2)]

    if img.dtype != np.uint8: # VERY WEIRD CODE OF MINE HERE IT REALLY MAKES NO SENSE JUST NOT CHANGING FOR COMPAT
        # just to remove the warning raised by img_as_ubyte
        # NB could do this per channel
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                # need manual conversion of the image so that it can be read as 8 bit or alike
                # force image between 0 and 1 then do convert
                if normalize:
                    img = img_as_ubyte((img - img.min()) / (
                            img.max() - img.min()))  # do I really need that --> probably better to do it myself ???
            except Exception:
                try:
                    img = img.astype(
                        np.uint8)  # error is probably due to a full black image --> can be fixed by converting it to uint8 dircetly!!!
                except Exception:
                    logger.error('error converting image to 8 bits')
                    return None

    # KEEP UNCHANGED
    # DIRTY HACK FOR BUG https://doc.qt.io/qtforpython-5/PySide2/QtGui/QImage.html Warning  Painting on a QImage with the format Format_Indexed8 is not supported.
    if len(img.shape) == 2:

        if False:
            # seomhow this code is buggy and I need the code below to fix it

            print('entering in there')

            img = np.stack([img, img, img], axis=-1)
            print(img.shape, img.dtype)
            if normalize:
                img = _normalize_8bits(img)

            # bytesPerLine = 3 * img.shape[-2]
            bytesPerLine = 3 * img.shape[-1]

            qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
                            QImage.Format_RGB888)  # --> bug fix but no display --> why ???
            return qimage
        else:

            # print('in there grabs', img.shape, img.max(), img.min())
            # already normalized
            # raise Exception('not good')
            if normalize:
                img = _normalize_8bits(img)
            return __numpy_to_qimage(img)

            #
            # print('entering in there')
            # # if normalize:
            # #     img = _normalize_8bits(img)
            # # bytes_per_line = 3 * img.shape[-2]
            # # qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytes_per_line, QImage.Format_Grayscale8)
            # # qimage = qimage.convertToFormat(QImage.Format_RGB888)
            # # return qimage
            # rgba_array = np.empty((img.shape[-2], img.shape[-1], 4), dtype=np.uint8)
            # rgba_array[:, :, 0] = img  # Red channel
            # rgba_array[:, :, 1] = img  # Green channel
            # rgba_array[:, :, 2] = img  # Blue channel
            # rgba_array[:, :, 3] = 255  # Alpha channel
            # img = rgba_array
            # if normalize:
            #     img = _normalize_8bits(img)
            # bytesPerLine = img.shape[-1] * img.shape[-1]
            #
            # qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
            #                 QImage.Format_ARGB32)  # --> bug fix but no display --> why ???
            # return qimage

    bytesPerLine = img.strides[1]

    if len(img.shape) == 3:
        nb_channels = img.shape[-1]
        logger.debug('Image has ' + str(nb_channels) + ' channels')

        if nb_channels == 3:
            # bug fix is here
            # https://stackoverflow.com/questions/55468135/what-is-the-difference-between-an-opencv-bgr-image-and-its-reverse-version-rgb-i

            if luts is not None:
                try:
                    img = blend_stack_channels_color_mode(img, luts=luts)
                except Exception:
                    pass

            if normalize:
                img = _normalize_8bits(img) # why I normalize after the blending ???  MAKES NO SENSE PROBABLY IF LUT I SHOULD NOT NORMALIZE AFTER!!!

            bytesPerLine = 3 * img.shape[-2]
            qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
                            QImage.Format_RGB888)  # --> bug fix but no display --> why ???
        elif nb_channels < 3:
            try:
                bgra = blend_stack_channels_color_mode(img, luts=luts)
            except Exception:
                bgra = np.zeros((img.shape[-3], img.shape[-2], 3), np.uint8, 'C')
                if img.shape[2] >= 1:
                    bgra[..., 0] = img[..., 0]
                if img.shape[2] >= 2:
                    bgra[..., 1] = img[..., 1]
                if img.shape[2] >= 3:
                    bgra[..., 2] = img[..., 2]
            bytesPerLine = 3 * bgra.shape[-2]
            if normalize:
                bgra = _normalize_8bits(bgra)
            qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_RGB888)
        else:
            # this does not make sense to handle ARGB in science but when I deactivate this all my images are black --> WHY
            # I really need to keep this for the alpha images of the mask but maybe still clean this code
            if nb_channels == 4 and preserve_alpha:
                bgra = np.zeros((img.shape[-3], img.shape[-2], 4), np.uint8, 'C')
                bgra[..., 0] = img[..., 0]
                bgra[..., 1] = img[..., 1]
                bgra[..., 2] = img[..., 2]
                if img.shape[2] >= 4:
                    logger.debug('using 4th numpy color channel as alpha for qimage')
                    bgra[..., 3] = img[..., 3]
                else:
                    bgra[..., 3].fill(255)

                bytesPerLine = 4 * bgra.shape[-2]
                qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
                                QImage.Format_ARGB32)
            else:
                try:
                    # allow display of the image as imageJ would do
                    # shall I do the same with 4 channels images ???
                    bgra = blend_stack_channels_color_mode(img, luts=luts)
                except Exception:
                    bgra = np.average(img,
                                      axis=-1)  # should I max it ??? ideally at some point I should use all the colors
                    bgra = np.stack([bgra, bgra, bgra], axis=-1).astype(np.uint8)

                if normalize:
                    bgra = _normalize_8bits(bgra) # again WHY NORMALIZE AFTER BLEND PROBABLY NEED NOT NORMALIZE IF BLEND/LUT
                bytesPerLine = 3 * bgra.shape[-2]

                qimage = QImage(bgra.data.tobytes(), bgra.shape[-2], bgra.shape[-3], bytesPerLine, QImage.Format_RGB888)

    else:
        '''KEEP MEGA IMPORTANT
        # https://doc.qt.io/qtforpython-5/PySide2/QtGui/QImage.html
        # Warning
        # Painting on a QImagewith the format Format_Indexed8 is not supported. --> NEVER USE THIS THAT'S IT AND FIX ALWAYS
        '''

        logger.warning('this should never have been reached --> errors may occur')
        # seg fault was due to inversion in width and height of the image --> need be careful and fix always
        qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], bytesPerLine,
                        QImage.Format_Indexed8)

    del img

    return qimage


def __numpy_to_qimage(array):
    from batoolset.settings.global_settings import set_UI  # set the UI to qtpy
    set_UI()
    from qtpy.QtGui import QImage  # mov
    height, width = array.shape

    if len(array.shape) == 2:
        # Convert grayscale image to 32-bit format with alpha channel
        rgba_array = np.empty((height, width, 4), dtype=np.uint8)
        rgba_array[:, :, 0] = array  # Red channel
        rgba_array[:, :, 1] = array  # Green channel
        rgba_array[:, :, 2] = array  # Blue channel
        rgba_array[:, :, 3] = 255  # Alpha channel
        array = rgba_array

    height, width, channels = array.shape
    bytes_per_line = channels * width

    if channels == 3:
        format = QImage.Format_RGB888
    elif channels == 4:
        format = QImage.Format_ARGB32
    else:
        raise ValueError("Unsupported number of channels")

    qimage = QImage(array.data.tobytes(), width, height, bytes_per_line, format)
    return qimage

def _get_white_bounds(img):
    # Find the coordinates where the image is not equal to 0 (white pixels)
    coords = np.where(img != 0)
    # Check if there are no white pixels in the image
    if coords[0].size == 0:
        # Return None to indicate no white pixels
        return None
    # Return the minimum and maximum row and column coordinates of the white pixels
    return np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])



def get_white_bounds(imgs):
    # Check if the input is a list of images
    if isinstance(imgs, list):
        # Initialize the bounds with large initial values
        bounds = [10000000, 0, 10000000, 0]
        # Iterate over each image in the list
        for img in imgs:
            # Get the bounds of white pixels in the current image
            curbounds = _get_white_bounds(img)
            # If there are no white pixels, continue to the next image
            if curbounds is None:
                continue
            # Update the overall bounds by taking the minimum and maximum values
            bounds[0] = min(curbounds[0], bounds[0])
            bounds[1] = max(curbounds[1], bounds[1])
            bounds[2] = min(curbounds[2], bounds[2])
            bounds[3] = max(curbounds[3], bounds[3])
        # Check if there were no white pixels in any of the images
        if bounds[0] == 10000000:
            # Return None to indicate no white pixels
            return None
    else:
        # Get the bounds of white pixels in the single image
        bounds = _get_white_bounds(imgs)
    # Return the overall bounds of white pixels
    return bounds



def crop_smartly(imgs, bounds=None, bounds_around=0):
    # Check if the bounds are not provided
    if bounds is None:
        # Calculate the bounds of white pixels in the images
        bounds = get_white_bounds(imgs)
        # Uncomment the following line to print the bounds
        # print(bounds)
    # Check if the input is a list of images
    if isinstance(imgs, list):
        # Iterate over each image in the list
        for iii, img in enumerate(imgs):
            # Crop the image based on the bounds and the specified additional padding
            imgs[iii] = img[bounds[0] - bounds_around:bounds[1] + bounds_around,
                            bounds[2] - bounds_around:bounds[3] + bounds_around]
        # Return the modified list of images
        return imgs
    else:
        # Crop the single image based on the bounds and the specified additional padding
        return imgs[bounds[0] - bounds_around:bounds[1] + bounds_around,
                    bounds[2] - bounds_around:bounds[3] + bounds_around]


def pad_border_xy(img, dim_x=-2, dim_y=-3, size=1, mode='symmetric', **kwargs):
    """
    Pad the borders of an image.

    Args:
        img (ndarray): The input image.
        dim_x (int, optional): The x-dimension index. Defaults to -2.
        dim_y (int, optional): The y-dimension index. Defaults to -3.
        size (int, optional): The padding size. Defaults to 1.
        mode (str or scalar, optional): The padding mode. Defaults to 'symmetric'.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The padded image.
    """

    # Check if size is non-positive, return the original image
    if size <= 0:
        return img

    # Check if the image is None, log an error, and return None
    if img is None:
        logger.error('Image is None -> nothing to do')
        return None

    smart_slices = []  # Initialize a list to store smart slices
    pad_seq = []  # Initialize a list to store padding sequences

    # Handle special cases where the dimension indices are negative and exceed the image shape
    if dim_y == -3 and abs(dim_y) > len(img.shape):
        if dim_x == -2:
            dim_x = -1
            dim_y = -2

    # Create slices and padding sequences for each dimension of the image shape
    for dim in range(len(img.shape)):
        smart_slices.append(slice(None))
        pad_seq.append((0, 0))

    # Set the smart slices and padding sequences for the x and y dimensions
    smart_slices[dim_x] = slice(size, -size)
    smart_slices[dim_y] = slice(size, -size)

    pad_seq[dim_x] = ((size, size))
    pad_seq[dim_y] = ((size, size))

    # Perform padding based on the mode specified
    if isinstance(mode, str):
        img = np.pad(img[tuple(smart_slices)], tuple(pad_seq), mode=mode)
    else:
        img = np.pad(img[tuple(smart_slices)], tuple(pad_seq), mode='constant', constant_values=mode)

    # Return the padded image
    return img


def blend(bg, fg, alpha=0.3, mask_or_forbidden_colors=None):
    """
    Blend foreground and background images with alpha transparency.

    Args:
        bg (ndarray): The background image.
        fg (ndarray): The foreground image.
        alpha (float, optional): The alpha value for blending. Defaults to 0.3.
        mask_or_forbidden_colors (ndarray or list, optional): The mask or list of forbidden colors. Defaults to None.

    Returns:
        ndarray: The blended image.
    """

    # Convert images to be compatible
    bg, fg = create_compatible_image(bg, fg, auto_sort=True)

    if bg.max() > 255 or bg.dtype != np.uint8:
        bg = np.interp(bg, (bg.min(), bg.max()), (0, 255))

    blended = fg * (alpha) + bg * (1. - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)

    if mask_or_forbidden_colors is None:
        return blended

    if not isinstance(mask_or_forbidden_colors, np.ndarray):
        mask_or_forbidden_colors = mask_colors(fg, colors_to_mask=mask_or_forbidden_colors)

    if mask_or_forbidden_colors is None:
        return blended

    bg, mask_or_forbidden_colors = create_compatible_image(bg, mask_or_forbidden_colors, auto_sort=False)
    if mask_or_forbidden_colors.dtype != bool:
        if mask_or_forbidden_colors.max() != 0:
            mask_or_forbidden_colors = mask_or_forbidden_colors / mask_or_forbidden_colors.max()

    bg = bg * mask_or_forbidden_colors
    bg = np.clip(bg, 0, 255)
    bg = bg.astype(np.uint8)
    blended[bg != 0] = bg[bg != 0]

    return blended


def blend_single_channel_images_n_mask_for_matplotlib(img, wing_mask, default_blend_channel_for_mask=0, alpha=0.3):
    # Quick n dirty code to blend an image and display it in matplotlib !!! just for display purpose
    # it assumes both the mask and img are single channel stuff

    tmp = np.zeros_like(img, shape=(*img.shape, 3))
    tmp[..., 0] = img
    tmp[..., 1] = img
    tmp[..., 2] = img


    wing_mask2 = np.zeros_like(wing_mask, shape=(*img.shape, 3), dtype=float)
    wing_mask2[..., default_blend_channel_for_mask] = wing_mask
    wing_mask2[wing_mask2 != 0] = tmp.max()

    bld = blend(tmp, wing_mask2, alpha=alpha, mask_or_forbidden_colors=0x000000)
    bld = bld / bld.max()
    return bld


# try apply a lut to that and then blend the RGB image --> much smarter in the end
def blend_stack_channels_color_mode(img, luts=None):
    """
    Blend stacked channels of an image into color mode using Look-Up Tables (LUTs).

    Args:
        img (ndarray): The input image with stacked channels.
        luts (list of ndarrays, optional): The list of LUTs for each channel. Defaults to None.

    Returns:
        ndarray: The blended image in color mode.
    """

    from batoolset.luts.lut_minimal_test import apply_lut
    from batoolset.luts.lut_minimal_test import PaletteCreator

    if len(img.shape) == 2:
        return np.stack([img, img, img], axis=-1)

    default_LUTs = ['RED', 'GREEN', 'BLUE', 'CYAN', 'MAGENTA', 'YELLOW', 'GRAY']
    lutcreator = PaletteCreator()

    final_image = np.zeros_like(img, shape=(*img.shape[:-1], 3), dtype=float)

    if luts is None:
        try:
            luts = img.metadata['LUTs']
        except Exception:
            pass

    for ch in range(img.shape[-1]):
        tmp = img[..., ch].astype(float)

        min_val = tmp.min()
        max_val = tmp.max()

        if min_val != max_val:
            tmp = (tmp - min_val) / (max_val - min_val)
        else:
            if max_val != 0:
                tmp = tmp / max_val

        if luts is not None:
            tmp = apply_lut(tmp, luts[ch], convert_to_RGB=True)
        else:
            try:
                lut = default_LUTs[ch]
            except Exception:
                lut = 'GRAY'
            lut = lutcreator.create3(lutcreator.list[lut])
            tmp = apply_lut(tmp, lut, convert_to_RGB=True)

        final_image = final_image + 1. / img.shape[-1] * tmp

    return final_image

# TODO maybe add range and dtype check at some point
# NB if auto --> match directly to the image with the max nb of channels/dimensions
# in fact need return just the image to be changed
# but need identify which image it is in fact --> need return an idx --> such as 0 or 1

# maybe fix image if it has more dims than 3
def create_compatible_image(desired_image, image_to_change, auto_sort=False):
    """
    Create a compatible image by modifying the shape or channels of the input image.

    Args:
        desired_image (ndarray): The desired image shape or channels.
        image_to_change (ndarray): The image to be modified.
        auto_sort (bool, optional): Automatically sort the images based on dimensions or channels. Defaults to False.

    Returns:
        tuple: The modified desired image and image to change to be compatible.

    Note:
        This function modifies the image_to_change to match the shape or channels of the desired_image.
        The returned images are not guaranteed to have the same dtype as the input images.
    """

    if len(desired_image.shape) >= 3 and desired_image.shape[-1] > 3:
        desired_image = np.average(desired_image, axis=-1)

    if len(desired_image.shape) == len(image_to_change.shape) and desired_image.shape == image_to_change.shape:
        return desired_image, image_to_change

    if abs(len(desired_image.shape) - len(image_to_change.shape)) > 1:
        logger.error('Images cannot be rendered compatible easily --> ignoring')
        return desired_image, image_to_change

    try:
        if not desired_image.shape[0] == image_to_change.shape[0] and desired_image.shape[1] == image_to_change.shape[1]:
            logger.error('Images cannot be rendered compatible easily because the two first dimensions are not equal between the two images')
            return desired_image, image_to_change
    except Exception:
        logger.error('Images cannot be rendered compatible easily because the two first dimensions are not equal between the two images')
        return desired_image, image_to_change

    need_swap = False
    desired_image_clone, image_to_change_clone = desired_image, image_to_change

    if auto_sort:
        if len(image_to_change.shape) > len(desired_image.shape):
            desired_image_clone, image_to_change_clone = image_to_change_clone, desired_image_clone
            need_swap = True
        elif len(image_to_change.shape) == len(desired_image.shape) and image_to_change.shape[-1] > desired_image.shape[-1]:
            desired_image_clone, image_to_change_clone = image_to_change_clone, desired_image_clone
            need_swap = True

    tmp = np.zeros_like(desired_image_clone, dtype=image_to_change_clone.dtype)

    for c in range(tmp.shape[-1]):
        tmp[..., c] = image_to_change_clone

    if not need_swap:
        return desired_image_clone, tmp
    else:
        return tmp, desired_image_clone


# not bad and can be useful !!!
# colors to mask should be an array colors/values
def mask_colors(colored_image, colors_to_mask, invert_mask=False, warn_on_color_not_found=False):
    """
    Create a mask based on specified colors in an image.

    Args:
        colored_image (ndarray): The colored image.
        colors_to_mask (list or tuple): The colors to be masked.
        invert_mask (bool, optional): Invert the mask. Defaults to False.
        warn_on_color_not_found (bool, optional): Log a warning if colors are not found. Defaults to False.

    Returns:
        ndarray: The mask with True values where colors are present, and False values elsewhere.
               None if colors are not found in the image.

    Note:
        The mask is created by checking if each pixel in the colored image matches any of the specified colors.
        The mask has the same shape as the colored image.

    """
    if colors_to_mask is None:
        logger.warning('No color to be masked was specified --> ignoring mask')
        return None

    if not (isinstance(colors_to_mask, list) or isinstance(colors_to_mask, tuple)):
        colors_to_mask = [colors_to_mask]

    if not invert_mask:
        mask = np.zeros_like(colored_image, dtype=bool)
    else:
        mask = np.ones_like(colored_image, dtype=bool)

    mask_value = True if not invert_mask else False

    for color in colors_to_mask:
        if len(mask.shape) == 3:
            mask[np.where(np.all(colored_image == color, axis=-1))] = mask_value
        else:
            mask[np.where(colored_image == color)] = mask_value

    if mask.max() == mask.min():
        if warn_on_color_not_found:
            logger.warning('Colors not found --> no mask created')
        return None

    return mask


# TODO maybe make this a more generic stuff
def get_nb_of_series_in_lif(lif_file_name):
    """
    Get the number of series (image sequences) in a .lif file.

    Args:
        lif_file_name (str): The path to the .lif file.

    Returns:
        int: The number of series in the .lif file.
        None if the file is not a .lif file or an error occurs.

    Note:
        This function uses the `read_lif` library to read the .lif file.

    """
    if not lif_file_name or not lif_file_name.lower().endswith('.lif'):
        logger.error('Error only lif file supported')
        return None

    reader = read_lif.Reader(lif_file_name)
    series = reader.getSeries()

    return len(series)

def get_series_count_in_lif(lif_file_name):
    """
    Get the number of series (image sequences) in a .lif file.

    Args:
        lif_file_name (str): The path to the .lif file.

    Returns:
        int: The number of series in the .lif file.
        None if the file is not a .lif file or an error occurs.

    Note:
        This function uses the `read_lif` library to read the .lif file.

    """
    if not lif_file_name or not lif_file_name.lower().endswith('.lif'):
        logger.error('Error only lif file supported')
        return None

    reader = read_lif.Reader(lif_file_name)
    series = reader.getSeries()

    for serie in series:
        # Additional information can be extracted here if needed
        # such as series name, time stamps, time lapse, dimensions, voxel sizes, etc.
        # Use the serie object methods to access the desired information

        # Example code to print series information
        print('name', serie.getName())
        print('ts', len(serie.getTimeStamps()))
        print('tl', serie.getTimeLapse())
        print('times', serie.getNbFrames())
        print('Zx ratio', serie.getZXratio())
        metadata = serie.getMetadata()
        print('voxel_size_x', metadata['voxel_size_x'])
        print('voxel_size_y', metadata['voxel_size_y'])
        print('voxel_size_z', metadata['voxel_size_z'])
        # dimensions = serie.getDimensions()
        # for dim in range(len(dimensions)):
        #     try:
        #         print('Voxel size dim', dim, serie.getVoxelSize(dimensions[dim])) # --> exactly what I want and need
        #     except:
        #         pass

    return len(series)

def get_series_names_in_lif(lif_file_name):
    """
    Get the names of series (image sequences) in a .lif file.

    Args:
        lif_file_name (str): The path to the .lif file.

    Returns:
        list: A list of series names in the .lif file.
        None if the file is not a .lif file or an error occurs.

    Note:
        This function uses the `read_lif` library to read the .lif file.

    """
    if not lif_file_name or not lif_file_name.lower().endswith('.lif'):
        logger.error('Error only lif file supported')
        return None

    reader = read_lif.Reader(lif_file_name)
    series = reader.getSeries()
    series_names = []

    for serie in series:
        series_names.append(serie.getName())
        # Additional information can be extracted here if needed
        # such as time stamps, time lapse, dimensions, voxel sizes, etc.
        # Use the serie object methods to access the desired information

    return series_names


@lru_cache(maxsize=256)
def pdf_page_count(path):
    """Number of pages of a PDF, read with PyMuPDF (AGPL) or, if it is not
    installed, pypdfium2 (Apache/BSD). ImportError if neither is installed."""
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(path) as doc:
            return doc.page_count
    try:
        import pypdfium2
    except ImportError as e:
        raise ImportError('PDF files need PyMuPDF (pip install pymupdf) '
                          'or pypdfium2 (pip install pypdfium2)') from e
    pdf = pypdfium2.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def get_series_info(filename):
    """
    Return (n_series, series_names) for files that support multiple series.
    Returns (1, None) for single-series or unsupported formats.

    Results are cached per filename (LRU, up to 256 entries) so repeated calls
    for the same file — e.g. from the annotation editor — do not re-read the
    file or restart a JVM.

    Args:
        filename (str): Path to the image file.

    Returns:
        tuple: (int, list|None) — number of series and list of names, or (1, None).
    """
    if not filename or not isinstance(filename, str):
        return 1, None

    # sidecar files (e.g. Leica .lifext) are not images — never probe them for series
    if is_forbidden_image_file(filename):
        return 1, None

    lower = filename.lower()

    # remote images (plain URLs, OME-Zarr, OMERO links) are one series each:
    # never hand a URL to Bio-Formats
    if lower.startswith(('http://', 'https://')):
        return 1, None

    # no such file (a PyFigures '@template@' slot, a moved image) or a folder
    # (local OME-Zarr): one series — Bio-Formats would only start its JVM to fail
    if not os.path.isfile(filename):
        return 1, None

    # a PDF (a panel): its pages are the "series"
    if lower.endswith('.pdf'):
        try:
            n = pdf_page_count(filename)
        except Exception:
            return 1, None
        return (n, [f'Page {i + 1}' for i in range(n)]) if n > 1 else (1, None)

    # Fast path: formats known to be single-series (no JVM needed).
    # .lsm is loaded by tifffile which does not expose Bio-Formats series.
    _single_series_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
                           '.lsm', '.xls', '.xlsx', '.lst', '.db', '.txt', '.svg')
    if lower.endswith(_single_series_exts):
        if lower.endswith(('.tif', '.tiff')):
            # a Fluoview/Micro-Manager TIFF can hold several positions (Fiji series)
            from batoolset.image_reader import tiff_positions
            n = tiff_positions(filename)
            if n > 1:
                return n, [f'Position {i + 1}' for i in range(n)]
        return 1, None

    # videos are single-series time-stacks — never route them through Bio-Formats/JVM
    if lower.endswith(VIDEO_EXTS):
        return 1, None

    # .lif files — use read_lif (pure Python, no JVM required)
    if lower.endswith('.lif'):
        try:
            names = get_series_names_in_lif(filename)
            if names and len(names) > 1:
                return len(names), names
        except Exception:
            pass
        return 1, None

    # Other formats: try Bio-Formats via IJ reader
    try:
        from batoolset.IJ.ij_reader import count_series, list_series
        n = count_series(filename)
        if n > 1:
            names = list_series(filename)
            return n, names
    except Exception:
        pass

    return 1, None


def _transfer_voxel_size_metadata(input_file_with_correct_metadata, output_file_with_missing_metadata):
    """
    Work in progress: Transfers voxel size metadata from one file to another.

    Args:
        input_file_with_correct_metadata (str): The path to the input file with correct metadata.
        output_file_with_missing_metadata (str): The path to the output file with missing metadata.

    Note:
        This function is a work in progress and requires further development.

    """
    img1 = Img(input_file_with_correct_metadata)
    img2 = Img(output_file_with_missing_metadata)

    relevant_metadatas = ['vx', 'vy', 'vz', 'AR', 'creation_time']

    meta_changed = False
    for relevant_metadata in relevant_metadatas:
        if relevant_metadata in img1.metadata:
            if relevant_metadata in img2.metadata:
                if img2.metadata[relevant_metadata] == img1.metadata[relevant_metadata]:
                    continue
                else:
                    img2.metadata[relevant_metadata] = img1.metadata[relevant_metadata]
                    meta_changed = True
            else:
                img2.metadata[relevant_metadata] = img1.metadata[relevant_metadata]
                meta_changed = True
        else:
            logger.warning('relevant metadata "' + relevant_metadata + '" not found --> ignoring')
            meta_changed = True

    if meta_changed:
        if 'vx' in img2.metadata and 'vy' in img2.metadata:
            # just to make resolution rationale
            tifffile.imwrite(output_file_with_missing_metadata, img2, imagej=False, resolution=(
                (int(1_000_000 / img2.metadata['vx']), 1000000), (int(1_000_000 / img2.metadata['vy']), 1000000)),
                             metadata=img2.metadata)
        else:
            tifffile.imwrite(output_file_with_missing_metadata, img2,
                             imagej=False, metadata=img2.metadata)


def mask_rows_or_columns(img, spacing_X=2, spacing_Y=None, masking_value=0, return_boolean_mask=False,
                         initial_shiftX=0, initial_shiftY=0, random_start=False):  # , dimension_h=-2, dimension_w=-1
    '''
    Creates lines where signal is removed --> can be used to train deep learning models à la noise2void.
    Be careful it creates an image out that is necessarily HWC --> even if original is not --> that may be counterintuitive.
    :param img: The input image to be masked.
    :param spacing_X: An integer specifying the horizontal interval at which lines of pixels will be masked.
                      This parameter is optional and defaults to 2.
    :param spacing_Y: An integer specifying the vertical interval at which lines of pixels will be masked.
                      This parameter is optional and defaults to None.
    :param masking_value: A value to be used for masking the input image. This parameter is optional and defaults to 0.
    :param return_boolean_mask: A boolean specifying whether to return the boolean mask instead of the masked image.
                                This parameter is optional and defaults to False.
    :param initial_shiftX: An integer specifying the initial shift value for the horizontal masking.
                           This parameter is optional and defaults to 0.
    :param initial_shiftY: An integer specifying the initial shift value for the vertical masking.
                           This parameter is optional and defaults to 0.
    :param random_start: A boolean specifying whether to use random initial shift values for masking.
                         This parameter is optional and defaults to False.
    :return: A masked image or a boolean mask depending on the value of `return_boolean_mask`.
    '''

    # If img is a tuple, create a boolean mask with the same shape as the tuple
    if isinstance(img, tuple):
        mask = np.zeros(img, dtype=bool)
    else:
        mask = np.zeros(img.shape, dtype=bool)

    # If the mask has less than three dimensions, add a new axis to the end of the shape
    if mask.ndim < 3:  # assume no channel so add one
        mask = mask[..., np.newaxis]

    # If the spacing_X parameter is less than or equal to 1, set it to None
    if spacing_X is not None:
        if spacing_X <= 1:
            spacing_X = None
    # If the spacing_Y parameter is less than or equal to 1, set it to None
    if spacing_Y is not None:
        if spacing_Y <= 1:
            spacing_Y = None

    # If random_start is True and initial_shiftX and initial_shiftY are 0, generate random values for them
    if initial_shiftX == 0 and initial_shiftY == 0 and random_start:
        if spacing_X is not None:
            initial_shiftX = random.randint(0, spacing_X)
        if spacing_Y is not None:
            initial_shiftY = random.randint(0, spacing_Y)

    # Iterate over the channels of the mask
    for c in range(mask.shape[-1]):
        # If spacing_Y is not None, set values in the mask at intervals of spacing_Y
        if spacing_Y is not None:
            if mask.ndim > 3:
                mask[..., initial_shiftY::spacing_Y, :, c] = True
            else:
                mask[initial_shiftY::spacing_Y, :, c] = True
        # If spacing_X is not None, set values in the mask at intervals of spacing_X
        if spacing_X is not None:
            mask[..., initial_shiftX::spacing_X, c] = True

    # If return_boolean_mask is True or img is a tuple, return the boolean mask
    if return_boolean_mask or isinstance(img, tuple):
        return mask

    # If the input image has less than three dimensions, add a new axis to the end of the shape
    if img.ndim < 3:  # assume no channel so add one
        img = img[..., np.newaxis]

    # Apply the mask to the input image by setting masked values to the masking_value
    img[mask] = masking_value

    # Return the masked image
    return img

def rescale(img, scaling_factor=2):
    return skimage.transform.rescale(img, 1. / scaling_factor, preserve_range=True, anti_aliasing=False, channel_axis=-1)

# TODO in development --> code that better and check whether it keeps the intensity range or not
def resize(img, new_size, order=1, preserve_range=False):
    '''
    Resizes an input image to a new size using the skimage.transform.resize function.
    :param img: The input image to be resized.
    :param new_size: A tuple specifying the new size of the image.
    :param order: An integer specifying the order of interpolation. This parameter is optional and defaults to 1.
    :param preserve_range: A boolean specifying whether to preserve the range of the input image.
                           This parameter is optional and defaults to False.
    :return: The resized image.
    '''
    from skimage.transform import resize
    # Use the skimage.transform.resize function to resize the input image to the new size
    img = resize(img, new_size, order=order, preserve_range=preserve_range)
    # Return the resized image
    return img

def __top_hat(image, type='black', structuring_element=footprint_rectangle((50, 50)), preserve_range=True):
    """
    Applies top hat transformation to an image.

    Args:
        image (ndarray): The input image.
        type (str): The type of top hat transformation to apply. Valid values are 'black' and 'white'.
        structuring_element (ndarray): The structuring element used for the operation.
        preserve_range (bool): Indicates whether to preserve the intensity range of the input image.

    Returns:
        ndarray: The resulting image after applying the top hat transformation.

    Note:
        There seems to be a bug in the white top hat function causing an infinite loop. Use with caution.

    """
    logger.debug('bg subtraction ' + str(type) + '_top_hat')
    try:
        if len(image.shape) == 4:
            out = np.zeros_like(image)
            for zpos, zimg in enumerate(image):
                for ch in range(zimg.shape[-1]):
                    out[zpos, ..., ch] = __top_hat_single_channel__(zimg[..., ch], type=type,
                                                                    structuring_element=structuring_element,
                                                                    preserve_range=preserve_range)
            return out
        elif len(image.shape) == 3:
            out = np.zeros_like(image)
            for ch in range(image.shape[-1]):
                out[..., ch] = __top_hat_single_channel__(image[..., ch], type=type,
                                                          structuring_element=structuring_element,
                                                          preserve_range=preserve_range)
            return out
        elif len(image.shape) == 2:
            out = __top_hat_single_channel__(image, type=type, structuring_element=structuring_element,
                                             preserve_range=preserve_range)
            return out
        else:
            print('Invalid shape --> ' + type + ' top hat failed, sorry...')
    except Exception:
        print(str(type) + ' top hat failed, sorry...')
        traceback.print_exc()
        return image


def black_top_hat(image, structuring_element=footprint_rectangle((50, 50)), preserve_range=True):
    """
    Applies black top hat transformation to an image.

    Args:
        image (ndarray): The input image.
        structuring_element (ndarray): The structuring element used for the operation.
        preserve_range (bool): Indicates whether to preserve the intensity range of the input image.

    Returns:
        ndarray: The resulting image after applying the black top hat transformation.

    """
    return __top_hat(image, type='black', structuring_element=structuring_element, preserve_range=preserve_range)

def white_top_hat(image, structuring_element=footprint_rectangle((50, 50)), preserve_range=True):
    """
    Applies white top hat transformation to an image.

    Args:
        image (ndarray): The input image.
        structuring_element (ndarray): The structuring element used for the operation.
        preserve_range (bool): Indicates whether to preserve the intensity range of the input image.

    Returns:
        ndarray: The resulting image after applying the white top hat transformation.

    """
    return __top_hat(image, type='white', structuring_element=structuring_element, preserve_range=preserve_range)


# somehow tophat does not work for 3D but why ???
def __top_hat_single_channel__(single_channel_image, type, structuring_element=footprint_rectangle((50, 50)), preserve_range=True):
    """
    Applies the top hat transformation to a single-channel image.

    Args:
        single_channel_image (ndarray): The input single-channel image.
        type (str): The type of top hat transformation to apply ('white' or 'black').
        structuring_element (ndarray): The structuring element used for the operation.
        preserve_range (bool): Indicates whether to preserve the intensity range of the input image.

    Returns:
        ndarray: The resulting image after applying the top hat transformation.

    """
    dtype = single_channel_image.dtype
    min_val = single_channel_image.min()
    max_val = single_channel_image.max()

    if type == 'white':
        out = white_tophat(single_channel_image, structuring_element)
    else:
        out = black_tophat(single_channel_image, structuring_element)

    if preserve_range and (out.min() != min_val or out.max() != max_val):
        out = out / out.max()
        out = (out * (max_val - min_val)) + min_val
        out = out.astype(dtype)

    return out


# ---------------------------------------------------------------------------
# Video support: videos are handled as time-stacks (xy-t), so a .mp4/.avi/.mkv
# behaves like a T (or Z) stack — the existing dimension-slider and projection
# machinery then works on it unchanged.
# ---------------------------------------------------------------------------
VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm')


# ---------------------------------------------------------------------------
# Companion / sidecar files that must never be opened as images. These sit next
# to a real image file and share a similar name, so file pickers and drag-and-drop
# happily list them, but they hold no readable pixel data of their own. Feeding
# them to Bio-Formats yields a bogus image with wrong dimensions instead of an
# error (e.g. Leica LAS X writes a `.lif` plus a `.lifext` metadata sidecar — only
# the `.lif` is a real image).
# ---------------------------------------------------------------------------
FORBIDDEN_IMAGE_EXTS = ('.lifext',)


def is_forbidden_image_file(path):
    """True for sidecar/companion files that look like images but cannot be opened
    as one (see FORBIDDEN_IMAGE_EXTS)."""
    return isinstance(path, str) and path.lower().endswith(FORBIDDEN_IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Image file extensions we can load — used to pick image files out of a folder
# (e.g. when a folder is dragged and dropped). Covers common raster formats,
# Bio-Formats microscopy formats, SVG vector, and videos (loaded as time-stacks).
# Companion sidecars in FORBIDDEN_IMAGE_EXTS are still excluded by is_image_file().
# ---------------------------------------------------------------------------
IMAGE_FILE_EXTS = (
    '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tga', '.svg', '.pdf',
    '.lsm', '.lif', '.czi', '.nd2', '.ims', '.oib', '.oif', '.zvi', '.dv',
    # more Bio-Formats microscopy formats (read through the Fiji reader)
    '.vsi', '.stk', '.ics', '.ids', '.oir', '.ndpi', '.svs', '.scn', '.mrc', '.r3d',
    '.lei', '.sld', '.btf',
    # formats Fiji opens (ImageJ / Bio-Formats readers): DICOM, Gatan, Image-Pro, ...
    '.dcm', '.dicom', '.dm3', '.dm4', '.ipl', '.seq', '.ipw', '.ipm', '.pic', '.fits', '.fit',
    '.fts', '.nrrd', '.nhdr', '.mha', '.mhd', '.nii', '.nii.gz', '.hdr', '.img', '.pgm',
    '.ppm', '.pbm', '.pnm', '.psd', '.pcx', '.pict', '.pct', '.jp2', '.j2k', '.webp', '.spe',
    '.sdt', '.lim', '.liff', '.xdce', '.mnc', '.am', '.top', '.avs', '.dib',
) + VIDEO_EXTS

def is_image_file(path):
    """True for files PyFigures can load as an image: the image formats Fiji
    opens (IMAGE_FILE_EXTS). Text files (.txt, .xml, .csv...) and companion
    sidecars never are."""
    if not isinstance(path, str):
        return False
    lower = path.lower()
    return lower.endswith(IMAGE_FILE_EXTS) and not is_forbidden_image_file(lower)


def list_image_files_in_folder(folder, recursive=True):
    """Return a naturally-sorted list of loadable image files inside *folder*.

    recursive=True walks sub-folders too (drop a folder -> get every image
    within). Returns [] if the folder holds no loadable images."""
    if not folder or not os.path.isdir(folder):
        return []
    results = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                if is_image_file(p):
                    results.append(p)
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and is_image_file(p):
                results.append(p)
    try:
        from natsort import natsorted
        return natsorted(results)
    except Exception:
        return sorted(results)


def is_video_file(path):
    return isinstance(path, str) and path.lower().endswith(VIDEO_EXTS)


# Two portable decoders/encoders, both pip wheels that bundle ffmpeg (Windows,
# macOS, Linux — no system ffmpeg needed): OpenCV (opencv-python) and PyAV
# (`pip install av`, which also brings the H.264 encoder). Neither is a hard
# dependency; video_backends() says what is there.
_VIDEO_INSTALL_HINT = "install PyAV (pip install av) — it bundles ffmpeg"


def video_backends():
    """{'cv2': bool, 'av': bool, 'h264': bool} — which video libraries can be
    imported, and whether an H.264 encoder (videos that play everywhere,
    including QuickTime / Safari / PowerPoint on a Mac) is available."""
    out = {'cv2': False, 'av': False, 'h264': False}
    try:
        import cv2  # noqa: F401
        out['cv2'] = True
    except Exception:
        pass
    try:
        import av
        out['av'] = True
        out['h264'] = 'libx264' in av.codecs_available or 'h264_mf' in av.codecs_available \
            or 'h264_videotoolbox' in av.codecs_available
    except Exception:
        pass
    return out


def _video_memory_budget():
    """Bytes a decoded video may take: a quarter of the free RAM, at most 4 GB."""
    budget = 4 * 1024 ** 3
    try:
        import psutil
        budget = min(budget, int(psutil.virtual_memory().available * 0.25))
    except Exception:
        pass
    return max(budget, 64 * 1024 ** 2)


def _frame_step(n_frames, h, w, max_bytes):
    """Keep every k-th frame so the decoded stack fits in *max_bytes*."""
    if not n_frames or not h or not w:
        return 1
    est = float(n_frames) * h * w * 3
    return max(1, int(np.ceil(est / max_bytes)))


def _read_video_cv2(path, max_bytes):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError('could not open video file ' + str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = _frame_step(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), max_bytes)
    frames = []
    i = 0
    # sequential read only: cap.set() seeks are wrong on duplicate-frame videos
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        i += 1
    cap.release()
    return frames, fps, step


def _read_video_av(path, max_bytes):
    import av
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else None
        step = _frame_step(stream.frames, stream.codec_context.height,
                           stream.codec_context.width, max_bytes)
        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0:
                frames.append(frame.to_ndarray(format='rgb24'))
    finally:
        container.close()
    return frames, fps, step


def read_video(path, max_bytes=None):
    """Decode a video file into a (T, H, W, C) uint8 stack (xy-t) plus metadata.

    Colour videos give (T, H, W, 3) in RGB order; genuinely grayscale videos
    collapse to (T, H, W). Returns (metadata, ndarray) — same contract as the
    other ImageReader branches, with fps stored in metadata['fps'].

    Decoded with OpenCV, or PyAV when OpenCV is missing or can't read the file.
    A movie too big for memory (*max_bytes*, default a quarter of the free RAM,
    at most 4 GB) keeps every k-th frame: metadata['frame_step'] = k and the fps
    is divided by k, so time stamps stay right."""
    if max_bytes is None:
        max_bytes = _video_memory_budget()
    have = video_backends()
    if not have['cv2'] and not have['av']:
        raise IOError('cannot read the video ' + str(path) + ': ' + _VIDEO_INSTALL_HINT)
    frames, fps, step, errors = [], None, 1, []
    for name, reader in (('cv2', _read_video_cv2), ('av', _read_video_av)):
        if not have[name]:
            continue
        try:
            frames, fps, step = reader(path, max_bytes)
        except Exception as e:
            errors.append(f'{name}: {e}')
            frames = []
        if frames:
            break
    if not frames:
        hint = '' if have['av'] else ' (' + _VIDEO_INSTALL_HINT + ' to read more formats)'
        raise IOError('no frames could be decoded from ' + str(path) + hint
                      + ('; ' + '; '.join(errors) if errors else ''))
    if step > 1:
        logger.warning('%s is too long to hold in memory: keeping 1 frame in %d' % (path, step))

    image = np.stack(frames, axis=0)  # (T,H,W,C) or (T,H,W)

    # collapse an RGB stack whose three channels are identical → true grayscale
    if image.ndim == 4 and image.shape[-1] == 3:
        if np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2]):
            image = image[..., 0]

    t = image.shape[0]
    h = image.shape[1]
    w = image.shape[2]
    c = image.shape[3] if image.ndim == 4 else None
    dims = 't' + 'hw' + ('c' if c else '')

    metadata = {'w': w, 'h': h, 'c': c, 'd': None, 't': t, 'bits': 8,
                'vx': None, 'vy': None, 'vz': None, 'AR': None,
                'dimensions': dims, 'LUTs': None, 'times': None,
                'Overlays': None, 'ROI': None, 'timelapse': None,
                'creation_time': None,
                'fps': fps / step if (fps and fps > 0) else None,
                'frame_step': step}
    return metadata, image


def _video_fourcc_for(path):
    import cv2
    lower = path.lower()
    if lower.endswith('.avi'):
        return cv2.VideoWriter_fourcc(*'MJPG')  # broadly compatible for .avi
    return cv2.VideoWriter_fourcc(*'mp4v')       # mp4 / mkv / mov / m4v / webm container


def _av_codec_for(path):
    """The PyAV encoder for *path*, or None (then OpenCV writes it): H.264 for
    mp4 / mov / m4v / mkv (plays everywhere), VP9 for webm; avi stays MJPG."""
    lower = path.lower()
    try:
        import av
    except Exception:
        return None
    if lower.endswith('.webm'):
        return 'libvpx-vp9' if 'libvpx-vp9' in av.codecs_available else None
    if lower.endswith(('.mp4', '.mov', '.m4v', '.mkv')):
        for c in ('libx264', 'h264_videotoolbox', 'h264_mf'):
            if c in av.codecs_available:
                try:
                    av.CodecContext.create(c, 'w')
                    return c
                except Exception:
                    continue
    return None


def _write_video_av(arr, path, fps, codec, crf):
    """Encode (frames, H, W[, C]) uint8 with PyAV: yuv420p (what players expect),
    so odd sizes are padded by one edge pixel."""
    import av
    from fractions import Fraction
    n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    pad_h, pad_w = h % 2, w % 2
    rate = Fraction(float(fps) if fps else 25.0).limit_denominator(1001)
    options = {'movflags': 'faststart'} if path.lower().endswith(('.mp4', '.mov', '.m4v')) else {}
    container = av.open(path, mode='w', options=options)
    try:
        stream = container.add_stream(codec, rate=rate)
        stream.width = w + pad_w
        stream.height = h + pad_h
        stream.pix_fmt = 'yuv420p'
        # one encoder thread: x264's worker threads crashed the process at
        # exit ~1 run in 15 when a Qt window had rendered the frames
        if codec == 'libx264':
            stream.options = {'crf': str(crf), 'preset': 'medium', 'threads': '1'}
        elif codec == 'libvpx-vp9':
            stream.options = {'crf': str(min(63, crf + 14)), 'b': '0', 'threads': '1'}
        else:
            stream.bit_rate = int(8 * (w * h) * float(rate) / 10)
        for i in range(n):
            frame = arr[i]
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=-1)
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] > 3:
                frame = frame[..., :3]
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format='rgb24')
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def write_video(array, path, fps=25.0, dimensions=None, max_width=3840, crf=17):
    """Write a time-stack (or z-stack) numpy array to a video file.

    *array* axes must resolve to (frames, H, W) or (frames, H, W, C). When the
    array carries a dimension string (via *dimensions* or Img metadata) the frame
    axis is taken from 't' if present, otherwise 'd'; any further extra axes are
    reduced by taking their first index. Non-uint8 data is min/max normalized.
    Width is capped at *max_width* (mp4 decoder limit — see project notes).

    With PyAV installed, mp4 / mov / m4v / mkv are H.264 (quality *crf*, lower is
    better) and webm is VP9 — they play everywhere; otherwise OpenCV writes the
    old MPEG-4 codec (avi is always MJPG)."""
    arr = np.asarray(array)

    if dimensions is None:
        dimensions = guess_dimensions(array)

    # move the frame axis (t preferred, else d) to axis 0, drop other extra axes
    if isinstance(dimensions, str) and dimensions:
        dims = dimensions
        frame_letter = 't' if 't' in dims else ('d' if 'd' in dims else None)
        if frame_letter is not None:
            arr = np.moveaxis(arr, dims.index(frame_letter), 0)
            dims = frame_letter + dims.replace(frame_letter, '')
            # reduce any remaining leading extra dims (before hw[c]) to index 0
            while len(dims) > (4 if dims.endswith('c') else 3):
                arr = arr[:, 0]
                dims = dims[0] + dims[2:]
    if arr.ndim < 3:
        raise ValueError('need at least 3 dimensions (frames, H, W) to write a video, got shape ' + str(arr.shape))

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        a = (a - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(a)
        arr = a.astype(np.uint8)

    n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    is_color = arr.ndim == 4 and arr.shape[-1] >= 3
    if w > max_width:
        logger.warning('video width %d exceeds %d px; players may fail to open it' % (w, max_width))

    _create_dir(path)
    codec = _av_codec_for(path)
    if codec:
        return _write_video_av(arr, path, fps, codec, crf)
    if path.lower().endswith(('.mp4', '.mov', '.m4v')):
        logger.warning('writing %s with the old MPEG-4 codec (no H.264 encoder found): it may '
                       'not play on a Mac or in a browser; %s' % (path, _VIDEO_INSTALL_HINT))
    try:
        import cv2
    except ImportError:
        raise IOError('cannot write the video ' + str(path) + ': ' + _VIDEO_INSTALL_HINT)
    writer = cv2.VideoWriter(path, _video_fourcc_for(path), float(fps) if fps else 25.0, (w, h), is_color)
    if not writer.isOpened():
        raise IOError('could not open video writer for ' + str(path))
    try:
        for i in range(n):
            frame = arr[i]
            if arr.ndim == 4:
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[-1] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[-1] == 1:
                    frame = frame[..., 0]
            writer.write(np.ascontiguousarray(frame))
    finally:
        writer.release()
    return path


class Img(np.ndarray):  # subclass ndarray


    # TODO allow load list of images all specified as strings one by one
    # TODO allow virtual stack --> open only one image at a time from a series, can probably do that with text files
    def __new__(cls, *args, t=0, d=0, z=0, h=0, y=0, w=0, x=0, c=0, bits=8, serie_to_open=None, dimensions=None,
                metadata=None, prefer_store_if_available=True, **kwargs) -> object:
        '''Creates a new instance of the Img class

        The image class is a numpy ndarray. It is nothing but a matrix of pixel values.

        Parameters
        ----------
        t : int
            number of time points of the image
        d, z : int
            number of z stacks of an image
        h, y : int
            image height
        w, x : int
            image width
        c : int
            number of color channels
        bits : int
            bits per pixel
        dimensions : string
            order and name of the dimensions of the image
        metadata : dict
            dict containing metadata entries and their corresponding values

        '''

        meta_data = cls._default_metadata()

        if metadata is not None:
            # if user specified some metadata update them
            meta_data.update(metadata)
        elif args and isinstance(args[0], Img):
            # recover old metadata from original image # is that the correct way
            try:
                meta_data.update(args[0].metadata)
            except Exception:
                pass

        img = None
        if len(args) == 1:
            arg = args[0]
            # case 1: Input array is an already an ndarray
            if isinstance(arg, np.ndarray):
                img = cls._from_ndarray(arg, meta_data, dimensions)
            elif isinstance(arg, (str, list)):
                logger.debug('loading ' + str(arg))
                # input is a string, i.e. a link to one or several files
                if isinstance(arg, list) or '*' in arg:
                    # series of images
                    img = cls._from_path_list(arg, meta_data)
                else:
                    # single image
                    img = cls._from_single_file(arg, meta_data, serie_to_open, prefer_store_if_available)
        else:
            # custom image creation : setting the dimensions
            img = cls._blank(meta_data, t=t, d=d, z=z, h=h, y=y, w=w, x=x, c=c, bits=bits)

        if img is None:
            raise ValueError(
                "Can't create Img: invalid arguments, file not supported or file does not exist ("
                + str(args) + ")")

        if not hasattr(img, 'z_scale'):
            # default z/d scaling factor (see get_d_scaling/set_d_scaling); without this a freshly
            # constructed Img raises AttributeError the first time get_d_scaling() is called
            img.z_scale = 1.0

        return img

    @staticmethod
    def _default_metadata():
        '''Returns a fresh default metadata dict (never share one dict instance between images)'''
        return {'dimensions': None,  # image dimensions
                'bits': None,  # bits per pixel
                'vx': None,  # voxel x size
                'vy': None,  # voxel y size
                'vz': None,  # voxel z size
                'AR': None,  # vz/vx ratio
                'LUTs': None,  # lut
                'cur_d': 0,  # current z/depth pos
                'cur_t': 0,  # current time
                'Overlays': None,  # IJ overlays
                'ROI': None,  # IJ ROIs
                'timelapse': None,  # time between frames when in a time lapse movie-->
                'creation_time': None,
                }

    @classmethod
    def _from_ndarray(cls, arr, meta_data, dimensions):
        '''Wraps an already existing ndarray (case 1 of __new__)'''
        img = np.asarray(arr).view(cls)
        img.metadata = meta_data
        if dimensions is not None:
            img.metadata['dimensions'] = dimensions
        return img

    @classmethod
    def _from_single_file(cls, path, meta_data, serie_to_open, prefer_store_if_available):
        '''Loads a single image file (case 2 of __new__), falling back to PIL if the
        central store/ImageReader can't handle the format'''
        creation_time = get_file_creation_time(path)
        img = None
        try:
            # if prefer_store_if_available: # see how I can handle that smartly... -−> not so easy

            if prefer_store_if_available and serie_to_open is None:
                # prefer reload the image from store if available --> will speed up the loading process and spare more resources
                # Skip the store when a specific series is requested: the store key is the filename
                # only (ignoring serie_to_open), so it would return the wrong series.
                meta, img = ImageCentralStore.load(path)  # may work

            if img is None:
                meta, img = ImageReader.read(path, serie_to_open=serie_to_open)
        except IndexError:
            raise       # a series that does not exist: say so, don't try PIL
        except Exception:
            img = None

        if img is None:
            from PIL import Image
            img = Image.open(path)

            meta = {
                'format': img.format,
                'dimensions': img.size,
                'mode': img.mode,
                'w': img.width,
                'h': img.height,
                # Add other metadata attributes as needed

            }
            mode = img.mode
            num_channels = {
                '1': 1,  # Black and white
                'L': 1,  # Grayscale
                'P': 1,  # Palette
                'RGB': 3,  # True color
                'RGBA': 4,  # True color with alpha channel
                'CMYK': 4,  # Cyan, Magenta, Yellow, and Key (Black)
                'YCbCr': 3,  # YCbCr color space
                'I': 1,  # Intensity
                'F': 1,  # Floating point intensity
            }.get(mode, None)
            if num_channels is not None:
                meta['c'] = num_channels

        meta_data.update(meta)
        meta_data['path'] = path  # add path to metadata
        # meta_data['creation_time'] = creation_time
        if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
            meta_data['creation_time'] = str(dt.datetime.now())
        img = np.asarray(img).view(cls)
        img.metadata = meta_data
        # img.metadata.update({'creation_time': creation_time})
        return img

    @classmethod
    def _from_path_list(cls, path_or_list, meta_data):
        '''Loads a series of images, either a glob pattern string or an explicit list
        of paths (case 3 of __new__)'''
        # TODO --> I need also implement addition of a list to the store -> TODO

        if isinstance(path_or_list, list):
            # the user directly provided a list --> just stack the images -−> can be useful --> there will be error if they don't have same size --> offer options for that but in a later step
            image_list = path_or_list
        else:
            image_list = [img for img in glob.glob(path_or_list)]
            image_list = natsorted(image_list)
        # maybe create an array of creation time in this case
        creation_time = []
        for file in image_list:
            creation_time.append(get_file_creation_time(file))
        img = ImageReader.imageread(image_list)  # TODO add metadata here too for w,h d and channels
        meta_data['path'] = path_or_list  ## add path to metadata # TODO make this an array of files instead --> smarter in a way
        # meta_data['creation_time'] = creation_time
        if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
            meta_data['creation_time'] = str(dt.datetime.now())
        img = np.asarray(img).view(cls)
        img.metadata = meta_data
        # img.metadata.update({'creation_time':creation_time})
        return img

    @classmethod
    def _blank(cls, meta_data, t=0, d=0, z=0, h=0, y=0, w=0, x=0, c=0, bits=8):
        '''Builds a new empty image from the requested dimensions (case 4 of __new__)'''
        dims = []
        dimensions = []
        if t != 0:
            dimensions.append('t')
            dims.append(t)
        if z != 0 or d != 0:
            dimensions.append('d')
            dims.append(max(z, d))
        if h != 0 or y != 0:
            dimensions.append('h')
            dims.append(max(h, y))
        if w != 0 or x != 0:
            dimensions.append('w')
            dims.append(max(w, x))
        if c != 0:
            dimensions.append('c')
            dims.append(c)

        dimensions = ''.join(dimensions)

        meta_data['dimensions'] = dimensions  # add dimensions to metadata
        dtype = np.uint8  # default is 8 bits
        if bits == 16:
            dtype = np.uint16  # 16 bits
        if bits == 32:
            dtype = np.float32  # 32 bits
        meta_data['bits'] = bits
        if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
            meta_data['creation_time'] = str(dt.datetime.now())
        img = np.asarray(np.zeros(tuple(dims), dtype=dtype)).view(cls)
        # array = np.squeeze(array) # TODO may be needed especially if people specify 1 instead of 0 ??? but then need remove some stuff
        # img = array
        img.metadata = meta_data
        return img

    def __array_finalize__(self, obj):
        '''Propagates custom attributes (metadata, z_scale) to new Img instances

        Called by numpy after __new__ in explicit construction, and also (without
        going through __new__) whenever a new Img is produced via view casting or
        slicing/ufuncs on an existing Img (e.g. img[0:10], img + 1, img.view(Img)).
        Without this, such derived arrays would have no metadata attribute at all
        and would raise AttributeError as soon as it is accessed.

        Note: the top-level metadata dict is shallow-copied so the new instance
        doesn't share/mutate the source's dict, and the known mutable-container
        entries ('LUTs', 'ROI', 'Overlays') are additionally deep-copied so that,
        e.g., appending an ROI on a crop doesn't silently also mutate the parent
        image's ROI list. Other entries are plain scalars (str/int/float/None)
        so a shallow copy is enough for them. This does not attempt to fix up
        entries (like 'dimensions') that may become stale after shape changing
        operations (e.g. reductions) -- that needs to be handled by the specific
        operation, not generically here.
        '''
        if obj is None:
            # we're inside __new__, which sets .metadata itself right after
            return

        metadata = getattr(obj, 'metadata', None)
        if metadata is not None:
            metadata = dict(metadata)
            for key in ('LUTs', 'ROI', 'Overlays'):
                if metadata.get(key) is not None:
                    metadata[key] = copy.deepcopy(metadata[key])
        self.metadata = metadata
        self.z_scale = getattr(obj, 'z_scale', 1.0)

    # TODO do implement it more wisely or drop it because it's simpler to access the numpy array directly...
    def get_pixel(self, *args):
        '''get pixel value

        TODO
        '''
        if len(args) == self.ndim:
            return self[tuple(args)]
        logger.critical('wrong nb of dimensions')
        return None

    # TODO do implement it more wisely or drop it because it's simpler to access the numpy array directly...
    def set_pixel(self, x, y, value):
        '''sets pixel value

        TODO
        '''
        # if len(args) == self.ndim:
        self[x, y] = value

    def get_dimension(self, dim):
        '''gets the specified image dimension length

        Parameters
        ----------
        dim : single char string
            dimension of interest

        Returns
        -------
        int
            dimension length
	    '''

        # force dimensions compatibility (e.g. use synonyms)
        if dim == 'z':
            dim = 'd'
        elif dim == 'x':
            dim = 'w'
        elif dim == 'y':
            dim = 'h'
        elif dim == 'f':
            dim = 't'

        if self.metadata['dimensions'] is None:
            logger.error('dimension ' + str(dim) + ' not found!!!')
            return None

        if dim in self.metadata['dimensions']:
            idx = self.metadata['dimensions'].index(dim)
            idx = idx - len(self.metadata['dimensions'])
            if self.ndim >= abs(idx) >= 1:
                return self.shape[idx]
            else:
                # logger.error('dimension ' + str(dim) + ' not found!!!')
                return None
        else:
            # logger.error('dimension ' + str(dim) + ' not found!!!')
            return None

    def get_dimensions(self):
        '''gets the length of all dimensions

        Returns
        -------
        dict
            a dict containing dimension name along with its length

        '''
        dimension_parameters = {}
        for d in self.metadata['dimensions']:
            dimension_parameters[d] = self.get_dimension(d)
        return dimension_parameters

    def get_dimensions_as_string(self):
        return self.metadata['dimensions']

    def get_dim_idx(self, dim):
        # force dimensions compatibility (e.g. use synonyms)
        if dim == 'z':
            dim = 'd'
        elif dim == 'x':
            dim = 'w'
        elif dim == 'y':
            dim = 'h'
        elif dim == 'f':
            dim = 't'
        if not dim in self.metadata['dimensions']:
            return None
        return self.metadata['dimensions'].index(dim)

    # TODO code this better -> pb I use this in colab --> not very smart
    # deprecated --> upe pop instead
    # def pop(self, pause=1, lut='gray', interpolation=None, show_axis=False, preserve_AR=True):
    #     '''pops up an image using matplot lib
    #
    #     Parameters
    #     ----------
    #     pause : int
    #         time the image should be displayed
    #
    #     interpolation : string or None
    #         interpolation for image display (e.g. 'bicubic', 'nearest', ...)
    #
    #     show_axis : boolean
    #         TODO
    #
    #     preserve_AR : boolean
    #         keep image AR upon display
    #
    #     '''
    #
    #     if self.ndim > 3:
    #         logger.warning("too many dimensions can't pop image")
    #         return
    #
    #     plt.ion()
    #     plt.axis('off')
    #     plt.margins(0)
    #
    #     plt.clf()
    #     plt.axes([0, 0, 1, 1])
    #
    #     ax = plt.gca()
    #     ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    #     ax.get_yaxis().set_visible(False)
    #     ax.margins(0)
    #
    #     if self.ndim == 3 and self.shape[2] <= 2:
    #         # create a 3 channel array from the 2 channel array image provided
    #         rgb = np.concatenate(
    #             (self[..., 0, np.newaxis], self[..., 1, np.newaxis], np.zeros_like(self[..., 0, np.newaxis])), axis=-1)
    #         with warnings.catch_warnings():
    #             warnings.simplefilter('ignore')
    #             plt.imshow(img_as_ubyte(rgb), interpolation=interpolation)
    #             # logger.debug("popping image method 1")
    #     else:
    #         if self.ndim == 2:
    #             # if image is single channel display it as gray instead of with a color lut by default
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter('ignore')
    #                 plt.imshow(img_as_ubyte(self), cmap=lut, interpolation=interpolation)  # self.astype(np.uint8)
    #                 # logger.debug("popping image method 2")
    #         else:
    #             # split channels if more than 3 channels maybe or remove the alpha channel ??? or not ??? see how to do that
    #             if self.shape[2] == 3:
    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter('ignore')
    #                     plt.imshow(img_as_ubyte(self),
    #                                interpolation=interpolation)
    #                     # logger.debug("popping image method 3")
    #             else:
    #                 for c in range(self.shape[2]):
    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter('ignore')
    #                         plt.imshow(img_as_ubyte(self[:, :, c]), cmap=lut, interpolation=interpolation)
    #                         if c != self.shape[2] - 1:
    #                             plt.show()
    #                             plt.draw()
    #                             plt.pause(pause)
    #                         # logger.debug("popping image method 4")
    #
    #     if not preserve_AR:
    #         ax.axis('tight')  # required to preserve AR but this necessarily adds a bit of white around the image
    #     ax.axis('off')
    #
    #     plt.show()
    #     plt.draw()
    #     plt.pause(pause)

    def _pad_border_xy(self, *args, **kwargs):
        # will that work ??? --> yes that works and that is relatively simple to implement --> maybe this is the way to proceed for all functions --> MEGA TODO
        self[...] = pad_border_xy(self, *args, **kwargs)[...]

    # @deprecated --> remove all of this soon --> use pad_border_xy instead
    def setBorder(self, distance_from_border_in_px=1, color=0):
        ''' Set n pixels at the border of the image to the defined color

        Parameters
        ----------
        distance_from_border_in_px : int
            Distance in pixels from the borders of the image.
        color : int or tuple
            new color (default is black = 0)
        '''
        logger.warning(
            'NB: setBorder is deprecated and will be removed soon --> use Img._pad_border_xy or pad_border_xy instead.')

        if distance_from_border_in_px <= 0:
            # ignore when distance < 0
            return

        val = color

        if self.has_c() and self.get_dimension('c') > 1 and not isinstance(color, tuple):
            # convert int color to tuple when tuple is required, i.e. when an img has several channels
            val = tuple([color] * self.get_dimension('c'))

        all_dims_before_hwc = []
        for d in self.metadata['dimensions']:  # keep all dimensions before hwc unchanged
            if d not in ['w', 'h', 'c', 'x', 'y']:
                all_dims_before_hwc.append(slice(None))

        # recolor the border
        for v in range(distance_from_border_in_px):
            all_dims_before_hwc.append(slice(None))
            all_dims_before_hwc.append(v)
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(v)
            all_dims_before_hwc.append(slice(None))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(-(v + 1))
            all_dims_before_hwc.append(slice(None))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(slice(None))
            all_dims_before_hwc.append(-(v + 1))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]

    # TODO in fact that is more complex I should not donwsample the channel or color dimension nor the time dimension --> so I need many more parameters and controls  --> quite good already but finalize that later
    def downsample(self, dimensions_to_downsample, downsampling_factor=2):
        '''Downsamples an image along the specified dimension by the specified factor

        Parameters
        ----------
        dimensions_to_downsample : string
            chars representing the dimension to downsample

        downsampling_factor : int
            downsampling factor

        Returns
        -------
        ndarray
            a downsampled image
        '''

        if downsampling_factor == 1:
            logger.error("downsampling with a factor = 1 means no downsampling, thereby ignoring...")
            return self

        if self.metadata['dimensions'] is None:
            logger.error("Image dimensions not specified!!!")
            return self
        idx = None

        for dim in self.metadata['dimensions']:
            if dim in dimensions_to_downsample:
                if idx is None:
                    idx = np.index_exp[::downsampling_factor]
                else:
                    idx += np.index_exp[::downsampling_factor]
            else:
                if idx is None:
                    idx = np.index_exp[:]
                else:
                    idx += np.index_exp[:]

        if idx is None:
            return self
        return self[idx]

    def rescale(self, factor=2):
        '''rescales an image (using scipy)

        Parameters
        ----------
        factor : int
            rescaling factor

        Returns
        -------
        ndarray
            a rescaled image

        '''
        return skimage.transform.rescale(self, 1. / factor, preserve_range=True, anti_aliasing=False, channel_axis=-1)

    def has_dimension(self, dim):

        '''Returns True if image has the specified dimension, False otherwise

        Parameters
        ----------
        dim : single char string
            dimension of interest

        Returns
        -------
        boolean
            True if dimension of interest exist in image
        '''

        # use dimension synonyms
        if dim == 'x':
            dim = 'w'
        if dim == 'y':
            dim = 'h'
        if dim == 'z':
            dim = 'd'
        if dim in self.metadata['dimensions']:
            return True
        return False

    def is_stack(self):
        '''returns True if image has a z/d dimension, False otherwise

        '''
        return self.has_d()

    def has_channels(self):
        '''returns True if image has a c dimension, False otherwise

        '''
        return self.has_c()

    def get_t(self, t):
        '''returns an image at time t, None otherwise

        Parameters
        ----------
        t : int
            time point of interest

        Returns
        -------
        ndarray
            image at time t or None
        '''
        if not self.is_time_series():
            return None
        if t < self.get_dimension('t'):  # TODO check code
            return self.imCopy(t=t)
        return None

    # set the current time frame
    def set_t(self, t):
        self.metadata['cur_t'] = t

    def get_d_scaling(self):
        '''gets the z/d scaling factor for the current image

        Returns
        -------
        float
            the depth scaling factor
        '''
        return self.z_scale

    def set_d_scaling(self, scaling_factor):
        '''sets the z/d scaling factor for the current image

        Parameters
        ----------
        scaling_factor : float
            the new image scaling factor

        '''
        self.z_scale = scaling_factor

    def has_t(self):
        '''returns True if the image is a time series, False otherwise

        '''
        return self.has_dimension('t')

    def is_time_series(self):
        '''returns True if the image is a time series, False otherwise

        '''
        return self.has_t()

    def has_d(self):
        '''returns True if the image is a Z-stack, False otherwise

        '''
        return self.has_dimension('d') or self.has_dimension('z')

    # check for the presence of LUTs
    def has_LUTs(self):
        return 'LUTs' in self.metadata and self.metadata['LUTs'] is not None

    # get LUTs
    def get_LUTs(self):
        if 'LUTs' in self.metadata:
            return self.metadata['LUTs']
        return None

    # set LUTs
    def set_LUTs(self, LUTs):
        self.metadata['LUTs'] = LUTs

    def has_c(self):
        '''returns True if the image has color channels, False otherwise

        '''
        return 'c' in self.metadata['dimensions']

    # mode can be IJ or raw --> if raw --> set IJ to false and save directly TODO clean the mode and mode is only for tif so far --> find a way to make it better and more optimal --> check also how mode would behave with z stacks, etc...
    def save(self, output_name, print_file_name=False, ijmetadata='copy', mode='IJ', compression=None):
        '''saves the current image

        Parameters
        ----------
        output_name : string
            name of the file to save
        compression : str or None
            lossless TIFF compression passed to tifffile.imwrite (e.g. 'zlib',
            'lzw'). Default None keeps images uncompressed (unchanged behaviour).
            Useful for binary/sparse masks where it shrinks files dramatically.

        '''

        if print_file_name:
            print('saving', output_name)

        if output_name is None:
            logger.error("No output name specified... ignoring...")
            return
        output_name = os.fspath(output_name)  # accept pathlib.Path too

        # videos out: write the time-stack (or z-stack) as a movie file
        if is_video_file(output_name):
            fps = 25.0
            try:
                meta_fps = self.metadata.get('fps') if hasattr(self, 'metadata') and self.metadata else None
                if meta_fps:
                    fps = float(meta_fps)
            except Exception:
                pass
            dims = None
            try:
                dims = guess_dimensions(self)
            except Exception:
                pass
            write_video(np.asarray(self), output_name, fps=fps, dimensions=dims)
            return

        # TODO maybe handle tif with stars in their name here to avoid loss of data but ok for now...
        if not '*' in output_name and (output_name.lower().endswith('.tif') or output_name.lower().endswith('.tiff')):
            _create_dir(output_name)
            if mode != 'IJ':  # TODO maybe do a TA mode or alike instead...
                out = self
                tifffile.imwrite(output_name, out, compression=compression)
            else:
                # create dir if does not exist
                out = self
                # apparently int type is not supported by IJ
                if out.dtype == np.int32:
                    out = out.astype(np.float32)  # TODO check if correct with real image but should be
                if out.dtype == np.int64:
                    out = out.astype(np.float64)  # TODO check if correct with real image but should be
                # IJ does not support bool type too
                if out.dtype == bool:
                    out = out.astype(np.uint8) * 255
                if out.dtype == np.double:
                    out = out.astype(np.float32)
                # if self.has_c():
                #     if not self.has_d() and self.has_t():
                #         out = np.expand_dims(out, axis=-1)
                #         out = np.moveaxis(out, -1, 1)
                #     out = np.moveaxis(out, -1, -3)
                #     tifffile.imwrite(output_name, out, imagej=True)  # make the data compatible with IJ
                # else:
                #     # most likely a big bug here --> fix it --> if has d and no t does it create a bug ???? --> maybe
                #     if not self.has_d() and self.has_t():
                #         out = np.expand_dims(out, axis=-1)
                #         out = np.moveaxis(out, -1, 1)
                #     out = np.expand_dims(out, axis=-1)
                #     # reorder dimensions in the IJ order
                #     out = np.moveaxis(out, -1, -3)
                #     tifffile.imwrite(output_name, out, imagej=True)  # this is the way to get the data compatible with IJ
                # should work better now and fix several issues... but need test it with real images
                # if image has no c --> assume all ok
                if self.metadata['dimensions'] is not None:
                    # print('in dims')
                    # print(self.has_c())  # why has no c channel ???
                    if not self.has_c():
                        out = out[..., np.newaxis]
                    if not self.has_d():
                        # Z goes AFTER time: a 'thw' movie must stay frames, not become slices
                        out = np.expand_dims(out, axis=1 if self.has_t() else 0)
                    if not self.has_t():
                        out = out[np.newaxis, ...]
                else:
                    # print('othyer')
                    # no dimension specified --> assume always the same order that is tzyxc --> TODO maybe ...tzyxc
                    if out.ndim < 3:
                        out = out[..., np.newaxis]
                    if out.ndim < 4:
                        out = out[np.newaxis, ...]
                    if out.ndim < 5:
                        out = out[np.newaxis, ...]

                # print('final', out.shape)

                out = np.moveaxis(out, -1, -3)  # need move c channel before hw (because it is default IJ style)

                # TODO maybe offer compression at some point to gain space ???
                # imageJ order is TZCYXS order with dtype is uint8, uint16, or float32. Is S a LUT ???? probably yes because (S=3 or S=4) must be uint8. can I use compression with ImageJ's Bio-Formats import function.
                # TODO add the possibility to save ROIs if needed...
                #        Parameters 'append', 'byteorder', 'bigtiff', and 'imagej', are passed             #         to TiffWriter(). Other parameters are passed to TiffWriter.save().
                # print(ijmetadata)

                # working version 2021.11.2

                ijmeta = {}
                if ijmetadata == 'copy':
                    if self.metadata['Overlays']:
                        ijmeta['Overlays'] = self.metadata['Overlays']
                    if self.metadata['ROI']:
                        ijmeta['ROI'] = self.metadata['ROI']
                    # TODO add support for Luts some day --> make sure IJ luts and epyseg lust are not incompatible or define an IJ_LUTs in metadata and get it
                    # make sure this does not create trouble
                    if self.metadata['LUTs']:
                        ijmeta['LUTs'] = self.metadata['LUTs']
                if not ijmeta:
                    ijmeta = None

                # old save code with deprecated ijmetadata
                if tifffile.__version__ < '2022.4.22':
                    tifffile.imwrite(output_name, out, imagej=True, ijmetadata=ijmeta,
                                     compression=compression,
                                     metadata={'mode': 'composite'} if self.metadata[
                                                                           'dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                else:
                    try:
                        # somehow this code doesn't seem to work with old tifffile but works with new one
                        from tifffile.tifffile import imagej_metadata_tag
                        # fix for ijmetadata deprecation in recent tifffile
                        ijtags = imagej_metadata_tag(ijmeta, '>') if ijmeta is not None else {}
                        # nb can add and save lut to the metadata --> see https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack

                        # quick hack to force images to display as composite in IJ if they have channels -> probably needs be improved at some point
                        tifffile.imwrite(output_name, out, imagej=True, metadata={'mode': 'composite'} if self.metadata[
                                                                                                              'dimensions'] is not None and self.has_c() else {},
                                         compression=compression,
                                         extratags=ijtags)  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                        # TODO at some point handle support for RGB 24-32 bits images saving as IJ compatible but skip for now
                        # nb tifffile.imwrite(os.path.join(filename0_without_ext,'tra_test_saving_24bits_0.tif'), tracked_cells_t0, imagej=True,                      metadata={}) --> saves as RGB if image RGB 3 channels

                        # TODO --> some day do the saving smartly with the dimensions included see https://pypi.org/project/tifffile/
                        # imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',  compression = 'deflate', planarconfig = 'separate', tile = (32, 32),    metadata = {'axes': 'TZCYX'})
                        # imwrite('temp.tif', volume, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),        metadata = {'spacing': 3.947368, 'unit': 'um', 'axes': 'ZYX'})
                    except Exception:
                        traceback.print_exc()
                        tifffile.imwrite(output_name, out, imagej=True, compression=compression,
                                         metadata={'mode': 'composite'} if self.metadata[
                                                                                                              'dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
        else:
            if output_name.lower().endswith('.zarr'):
                try:
                    import zarr as _zarr
                except ImportError:
                    raise ImportError("zarr is required to save .zarr files. Install with: pip install zarr")
                _create_dir(output_name)
                _store = _zarr.open(output_name, mode='a')
                _dims = self.metadata.get('dimensions') if self.metadata else None
                # choose chunk shape: one Z-plane at a time for 3-D, else auto
                _chunk = (1, self.shape[-2], self.shape[-1]) if self.ndim == 3 else True
                if 'mask' in _store:
                    del _store['mask']
                _store.create_dataset('mask', data=np.asarray(self), chunks=_chunk,
                                      dtype=self.dtype)
                # persist metadata as JSON attributes on the store
                if _dims is not None:
                    _store.attrs['dimensions'] = _dims
                if self.metadata:
                    for _k in ('bits', 'vx', 'vy', 'vz', 'AR', 'd', 'h', 'w', 'c', 't'):
                        if _k in self.metadata and self.metadata[_k] is not None:
                            _store.attrs[_k] = self.metadata[_k]
                return
            if output_name.lower().endswith('.npy') or output_name.lower().endswith('.epyseg'):
                # directly save as .npy --> the numpy default array format
                _create_dir(output_name)
                np.save(output_name, self,
                        allow_pickle=False)  # set allow pickle false to avoid pbs as pickle is by def not stable

                if self.metadata is not None and 'times' in self.metadata.keys():
                    times = self.metadata['times']
                    # force serialisation of times
                    self.metadata['times'] = str(times)
                    with open(output_name + '.meta', 'w', encoding='utf-8') as outfile:
                        json.dump(self.metadata, outfile)

                    # restore time metadata
                    self.metadata['times'] = times
                    # print('exporting metadata', self.metadata)  # metadata is not set --> too bad --> why
                # np.savez_compressed(output_name, self ) allow_pickle=False {'allow_pickle':False} --> maybe pass that
                return

            # the huge pb with this is that it is not portable --> because it necessarily uses pickle --> very dangerous save and too bad cause would allow saving metadata easily if passed as an array...
            if output_name.lower().endswith('.npz'):
                # directly save as .npy --> the numpy default array format
                _create_dir(output_name)
                # VERY GOOD IDEA TODO data is saved as data.npy inside the npz --> could therefore also save metadata ... --> VERY GOOD IDEA
                np.savez_compressed(output_name,
                                    data=self)  # set allow pickle false to avoid pbs as pickle is by def not stable
                return

            if not '*' in output_name and (self.has_t() or self.has_d()):
                logger.warning(
                    "image is a stack and cannot be saved as a single image use a geneic name like /path/to/img*.png instead")
                return
            else:
                _create_dir(output_name)
                if not self.has_t() and not self.has_d():
                    new_im = Image.fromarray(self)
                    new_im.save(output_name)
                    self.save_IJ_ROIs_or_overlays(output_name)
                    # try save IJ ROIs and overlays if they exist
                else:
                    # TODO recode below to allow any number of dimensions
                    if self.has_t():
                        t_counter = 0
                        # loop over all times of the image
                        for t in self[:]:
                            z_counter = 0
                            # loop over all z of the image
                            for z in t[:]:
                                if z.ndim == 3 and z.shape[2] <= 2:
                                    # create a 3 channel array from the 2 channel array image provided
                                    z = np.concatenate((z[..., 0, np.newaxis], z[..., 1, np.newaxis],
                                                        np.zeros_like(z[..., 0, np.newaxis])), axis=-1)
                                with warnings.catch_warnings():  # force it to be 8 bits for these formats
                                    warnings.simplefilter('ignore')
                                    z = img_as_ubyte(z)
                                new_im = Image.fromarray(z)
                                new_im.save(output_name.replace('*', 't{:03d}_z{:04d}'.format(t_counter,
                                                                                              z_counter)))  # replace * by tover 3 digit and z over 4 digits
                                z_counter += 1
                            t_counter += 1
                        self.save_IJ_ROIs_or_overlays(output_name)
                    elif self.has_d():
                        # loop over all z of the image
                        z_counter = 0
                        for z in self[:]:
                            if z.ndim == 3 and z.shape[2] <= 2:
                                # create a 3 channel array from the 2 channel array image provided
                                z = np.concatenate((z[..., 0, np.newaxis], z[..., 1, np.newaxis],
                                                    np.zeros_like(z[..., 0, np.newaxis])), axis=-1)
                            with warnings.catch_warnings():  # force it 8 bits for these rough formats
                                warnings.simplefilter('ignore')
                                z = img_as_ubyte(z)
                            new_im = Image.fromarray(z)
                            new_im.save(
                                output_name.replace('*', 'z{:04d}'.format(z_counter)))  # replace * by z over 4 digits
                            z_counter += 1
                        self.save_IJ_ROIs_or_overlays(output_name)

    # returns IJ ROIs from metadata
    def get_IJ_ROIs(self):
        try:
            # trying to save ROIs from ij images
            from roifile import ImagejRoi
            rois = []
            if self.metadata['Overlays'] is not None:
                overlays = self.metadata['Overlays']
                if isinstance(overlays, list):
                    if overlays:
                        overlays = [ImagejRoi.frombytes(roi) for roi in overlays]
                        rois.extend(overlays)
                else:
                    overlays = ImagejRoi.frombytes(overlays)
                    rois.append(overlays)
            if self.metadata['ROI'] is not None:
                rois_ = self.metadata['ROI']
                print(len(rois_), rois_)
                if isinstance(rois_, list):
                    if rois_:
                        rois_ = [ImagejRoi.frombytes(roi) for roi in rois_]
                        rois.extend(rois_)
                else:
                    rois_ = ImagejRoi.frombytes(rois_)
                    rois.append(rois_)
            if not rois:
                return None

            return rois
        except Exception:
            # no big deal if it fails --> just print error for now
            traceback.print_exc()

    # maybe do an IJ ROI editor some day ????
    # saves IJ ROIs as a .roi file or .zip file
    def save_IJ_ROIs_or_overlays(self, filename):
        """
        Saves ImageJ ROIs or overlays to a file.

        Args:
            filename (str): The filename for the output file.

        Returns:
            None

        """
        try:
            rois = self.get_IJ_ROIs()
            if not rois:
                return

            output_filename = filename

            if len(rois) > 1:
                output_filename += '.zip'
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            else:
                output_filename += '.roi'

            if rois is not None and rois:
                for roi in rois:
                    roi.tofile(output_filename)

        except Exception:
            traceback.print_exc()

    def get_width(self):
        """
        Get the width of the image.

        Returns:
            int: The width of the image.

        """
        return self.get_dimension('w')

    def get_height(self):
        """
        Get the height of the image.

        Returns:
            int: The height of the image.

        """
        return self.get_dimension('h')

    def projection(self, type='max'):
        """
        Perform a projection of the image along one or more dimensions.

        Args:
            type (str): The type of projection. Default is 'max'.

        Returns:
            Img: The projected image.

        """
        proj_dimensions = []
        if self.has_t():
            proj_dimensions.append(self.get_dimension('t'))
        proj_dimensions.append(self.get_height())
        proj_dimensions.append(self.get_width())
        if self.has_c():
            proj_dimensions.append(self.get_dimension('c'))

        projection = np.zeros(tuple(proj_dimensions), dtype=self.dtype)

        if type == 'max':
            if self.has_t():
                # Perform projection for each channel
                if self.has_c():
                    for t in range(self.shape[0]):
                        if self.has_d():
                            for z in self[t][:]:
                                for i in range(z.shape[-1]):
                                    projection[t, ..., i] = np.maximum(projection[t, ..., i], z[..., i])
                    return Img(projection, dimensions='thwc')
                else:
                    for t in range(self.shape[0]):
                        if self.has_d():
                            for z in self[t]:
                                projection[t] = np.maximum(projection[t], z)
                    return Img(projection, dimensions='thw')
            elif self.has_c():
                if self.has_d():
                    for z in self[:]:
                        for i in range(z.shape[-1]):
                            projection[..., i] = np.maximum(projection[..., i], z[..., i])
                return Img(projection, dimensions='hwc')
            else:
                if self.has_d():
                    for z in self[:]:
                        projection = np.maximum(projection, z)
                return Img(projection, dimensions='hw')
        else:
            logger.critical("Projection type " + type + " is not supported yet")
            return None
        return self

    # RETRY NOTE (2026-07-23): the DANGER warning below is from an earlier attempt at
    # overriding __str__ that "messed up the whole class and slicing" for reasons that
    # were never root-caused. Best guess: at the time, Img had no __array_finalize__, so
    # numpy's own array-printing code -- which internally creates summarized edge-slices
    # of the array (new Img instances) when formatting anything past a small size -- would
    # produce slices with NO .metadata attribute at all. Since the old __str__ touched
    # self.metadata directly, printing/logging a sufficiently large or sliced Img anywhere
    # (including via a stray debug print deep inside some unrelated slicing/tiling code)
    # would raise AttributeError from inside numpy's own printing internals, surfacing as
    # unrelated-looking breakage. __array_finalize__ was added this session (see above),
    # which should prevent that specific failure mode, so __repr__/__str__ are being
    # reintroduced -- written defensively (never raises, falls back to the default ndarray
    # repr) as a safety margin.
    # REVERT: if anything breaks, delete __repr__ and __str__ just below (keep to_string(),
    # it is unaffected and predates this). A pre-refactor backup of the whole file is at
    # batoolset/img.py.bak_20260722_084338. See memory note "img_repr_str_retry" for details.
    def __repr__(self):
        '''Short one-line repr for debugging (shape/dtype/dimensions/path). Deliberately
        never raises -- falls back to the default ndarray repr on any failure, see the
        RETRY NOTE above for why that matters here.
        '''
        try:
            meta = getattr(self, 'metadata', None) or {}
            extra = ''
            dims = meta.get('dimensions')
            if dims is not None:
                extra += ', dimensions=' + repr(dims)
            path = meta.get('path')
            if path is not None:
                extra += ', path=' + repr(path)
            return 'Img(shape=' + str(self.shape) + ', dtype=' + str(self.dtype) + extra + ')'
        except Exception:
            return super().__repr__()

    def __str__(self):
        return self.__repr__()

    # TODO DANGER!!!! OVERRIDING __str__ CAUSES HUGE TROUBLE BUT NO CLUE WHY
    #  --> this messes the whole class and the slicing of the array --> DO NOT PUT IT BACK --> NO CLUE WHY THOUGH
    # def __str__(self):
    def to_string(self):
        """
        Convert the image object to a string representation.

        Returns:
            str: The string representation of the image.

        """
        description = '#' * 20
        description += '\n'
        description += 'Image:'
        description += '\n'
        description += 'vx=' + str(self.metadata['vx']) + ' vy=' + str(self.metadata['vy']) + ' vz=' + str(
            self.metadata['vz'])
        description += '\n'
        description += 'dimensions=' + self.metadata['dimensions']
        description += '\n'
        description += 'shape=' + str(self.shape)
        description += '\n'
        description += self.metadata.__str__()
        description += '\n'
        dimensions_sizes = self.get_dimensions()
        for k, v in dimensions_sizes.items():
            description += k + '=' + str(v) + ' '
        description += '\n'
        description += str(super.__str__(self))
        description += '\n'
        description += '#' * 20
        return description

    # should dynamically crop images
    def crop(self, **kwargs):
        '''crops an image

        Parameters
        ----------
        kwargs : dict
            a dict containing the top left corner and the bottom right coordinates of the crop x1, y1, x2, y2

        Returns
        -------
        ndarray
            a crop of the image
        '''
        img = self
        corrected_metadata = dict(self.metadata)
        dims = []
        for i in range(len(img.shape)):
            dims.append(slice(None))

        # get the dim and its begin and end and create the appropriate slice
        for key, value in kwargs.items():
            if key in self.metadata['dimensions']:
                idx = self.metadata['dimensions'].index(key)
                if isinstance(value, list):
                    if len(value) == 2:
                        dims[idx] = slice(value[0], value[1])
                    elif len(value) == 3:
                        dims[idx] = slice(value[0], value[1], value[2])
                        # update the width and height parameters then or suppress w and h parameters from the data to avoid pbs
                    elif len(value) == 1:
                        dims[idx] = value
                        corrected_metadata.update(
                            {'dimensions': corrected_metadata['dimensions'].replace(key, '')})  # do remove dimension
                else:
                    if value is not None:
                        dims[idx] = value
                        corrected_metadata.update(
                            {'dimensions': corrected_metadata['dimensions'].replace(key, '')})  # do remove dimension
                    else:
                        dims[idx] = slice(None)
                # TODO need reduce size dim for the stuff in the metadata to avoid bugs

        img = np.ndarray.copy(img[tuple(dims)])
        output = Img(img, metadata=corrected_metadata)
        return output

    # should be able to parse any dimension in fact by its name
    # IMPORTANT NEVER CALL IT COPY OTHERWISE OVERRIDES THE DEFAULT COPY METHOD OF NUMPY ARRAY THAT CREATES ERRORS
    def imCopy(self, t=None, d=None, c=None):
        '''

        Parameters
        ----------
        t : int
            the index of the time series to copy

        d : int
            the index of the z/d to copy

        c : int
            the channel to copy

        Returns
        -------
        Img
            a (sub)copy of the image
        '''
        img = self

        corrected_metadata = dict(self.metadata)

        dims = []
        for i in range(len(img.shape)):
            dims.append(slice(None))

        if t is not None and self.has_t():
            idx = self.metadata['dimensions'].index('t')
            dims[idx] = t
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('t', '')})
        if d is not None and self.has_d():
            idx = self.metadata['dimensions'].index('d')
            dims[idx] = d
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('d', '')})
        if c is not None and self.has_c():
            idx = self.metadata['dimensions'].index('c')
            dims[idx] = c
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('c', '')})

        # TODO finalize this to handle any slicing possible --> in fact it's relatively easy

        img = np.ndarray.copy(img[tuple(dims)])
        output = Img(img, metadata=corrected_metadata)
        return output

    def within(self, x, y):
        ''' True if a pixel within the image, False otherwise

        '''
        if x >= 0 and x < self.get_width() and y >= 0 and y < self.get_height():
            return True
        return False

# ImageReader was moved to batoolset/image_reader.py on 2026-07-23 (one of the largest
# self-contained chunks of this file). Re-exported here so existing code doing
# `from batoolset.img import ImageReader` keeps working unchanged.
from batoolset.image_reader import ImageReader

if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    if False:
        # just a basic test for the newly added zarr stuff
        img = Img('C:/Users/aigouy/OneDrive/Desktop/test_images/sample_images_PA/handCorrection.zarr')
        print(img.shape)
        print(img.metadata)

        for z in range(img.shape[0]):
            plt.imshow(img[z])
            plt.show()

        sys.exit(0)

    if True:
        # FILE = '/home/aigouy/Bureau/sample_images_FIJI/tst_ROI_in_someZ_positions.tif'
        # FILE = '/home/aigouy/Bureau/sample_images_FIJI/sample2.lsm'
        # FILE = '/home/aigouy/Bureau/sample_images_FIJI/new-lenna.jpg'
        # FILE = '/F/Confocal_le_bivic/180920_overnight_tila_EcadKI_16_17hAPF_at_25degrees/predict/Image148.tif'
        FILE = '/home/aigouy/Bureau/sample_images_FIJI/mouse-kidney.lif'

        # that works but maybe make it less verbose
        tmp = Img(FILE, serie_to_open=2)
        print(tmp.shape)
        print(tmp.max(), tmp.min())
        print(tmp.metadata)
        tst = auto_scale(Img(FILE))
        print((tst.max()), tst.min())
        print(tst.shape)
        sys.exit(0)


    if True:
        from batoolset.tools.logger import TA_logger


        print('bob')
        sys.exit(0)


    if True:
        # tst = Img('/media/teamPrudhomme/EqpPrudhomme2/Vani/RNAi/Transvection_WT/Het/RNAi_het_20240312/26296_het_F1.tif')
        # tst = Img('/media/teamPrudhomme/EqpPrudhomme2/Vani/RNAi/Transvection_WT/Het/RNAi_het_20240312/32458_het_F1.tif') # good --> may need to fix pixel size auto or to take a frequency cut off in case of hot pixels -−> TODO
        # tst = Img('/E/Gautham/wing_feature_segmenter/results/real_tests_fluo_bernard_2/bckup_for_quantif_comparison/deletions_intron_F1b_F1ab_F1ae_10kb_Cherry/10kb AttpX1 Delta F1b Female HM0005.tif') # good --> may need to fix pixel size auto or to take a frequency cut off in case of hot pixels -−> TODO--> THIS IS FULLY SATURATED --> ADD IT TO THE PIPELINE
        # tst = Img('/E/Gautham/wing_feature_segmenter/results/real_tests_fluo_bernard_2/bckup_for_quantif_comparison/deletions_intron_F1b_F1ab_F1ae_10kb_Cherry/10 kb Cherry attpX1 Delta F1ab-1 Tg1 Male.tif') # good --> may need to fix pixel size auto or to take a frequency cut off in case of hot pixels -−> TODO--> THIS IS FULLY SATURATED --> ADD IT TO THE PIPELINE
        tst = Img('/E/Gautham/wing_feature_segmenter/results/real_tests_fluo_bernard_2/bckup_for_quantif_comparison/mutations_Spot196_10kb_Cherry/Spot196 mut15 Tg1 Male 0003.tif') # mega saturated --> cannot conclude anything
        plt.imshow(get_saturated_pixels_mask(tst, max=None)) # if max is known then set it
        # plt.imshow(get_saturated_pixels_mask(Img('/media/teamPrudhomme/EqpPrudhomme2/Vani/RNAi/Transvection_WT/Het/HET/28669_het_F1.tif'), max=90), cmap='gray') # if max is known then set it
        # plt.imshow(get_saturated_pixels_mask(Img('/media/teamPrudhomme/EqpPrudhomme2/Vani/RNAi/Transvection_WT/Het/HET/28669_het_F1.tif'), max=90)) # if max is known then set it
        plt.show()
        sys.exit(0)

    if True:
        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)
        pop(img[10:100, 10:100])
        sys.exit(0)

    if False:
        img = Img('https://samples.fiji.sc/new-lenna.jpg')[..., 0]
        tst = convolve(img, kernel=[[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
        plt.imshow(tst)
        plt.show()

    if True:
        # ok TODO --> do a master cleaning some day
        img = Img('/E/Sample_images/sample_images_PA/mini_asym/*.png')
        print(img.shape)
        import sys

        sys.exit(0)

    if True:
        import timeit

        img = Img('/E/Sample_images/clara_tests_3D_mesh/E9 WT GM130V actinR 4.lsm')
        print(img.metadata)
        print(img.shape)

        print(timeit.timeit(lambda: Img('/E/Sample_images/clara_tests_3D_mesh/E9 WT GM130V actinR 4.lsm'),
                            number=50))  # -->7.5 secs with meta
        import sys

        sys.exit(0)

    if False:
        # open image url
        # img = Img('https://samples.fiji.sc/new-lenna.jpg')
        # img = Img('file:///E/Sample_images/sample_images_PA/mini/focused_Series012.png')
        img = Img('file:///home/aigouy/Bureau/12_9.tif')
        print(img.metadata)
        print(img.shape)
        import sys

        sys.exit(0)

    if True:
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_2.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_noROI.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_noROI_channels_changed.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_ROIs_n_LUTs.tif'

        # bug ROI is not there but why ???
        # marche pas mais pkoi ????
        img = Img(f)
        print(img.shape)
        Img(img, dimensions='dhwc').save('/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/test.tif')

        import sys

        sys.exit(0)

    if True:
        # now epyseg reads the image properly but IJ does not for the voxel size --> need hack it a bit in order to get the stuff done properly
        img = Img('/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif')

        print(img.max(), img.min())  # 1.7492598 -0.0020951158 --> ok
        # see the RGB version of the image and maybe get that all the time

        SQL_plot = Img(
            '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000/tracked_cells_resized.tif')

        # img = img*255 + 1 # that fixes the bug --> see how I can do it in a cleaner and generic way

        composite = blend(img, SQL_plot, alpha=0.3, mask_or_forbidden_colors=0x000000)

        print(img.shape)
        print(SQL_plot.shape)
        print(composite.shape)

        print(img.dtype)
        print(SQL_plot.dtype)
        print(composite.dtype)

        print(img.max(), img.min())  # 1.7492598 -0.0020951158
        print(SQL_plot.max(), SQL_plot.min())  # 255 0
        print(composite.max(), composite.min())  # 77 0

        # in fact the original image should be between 0 and 255 most likely

        plt.imshow(composite)
        plt.show()
        # plt.imshow(img)
        # plt.show()

        import sys

        sys.exit(0)

    data = np.zeros((1024, 1024), dtype=np.uint8)

    if True:
        # now epyseg reads the image properly but IJ does not for the voxel size --> need hack it a bit in order to get the stuff done properly
        import sys

        sys.exit(0)
