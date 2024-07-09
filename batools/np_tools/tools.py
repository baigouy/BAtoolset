# here I will put numpy tips
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy as np

def get_value_from_nd_coords(img, nd_coords):
    """
    Retrieves the value from an image at the specified N-dimensional coordinates.

    Args:
        img (numpy.ndarray): The input image.
        nd_coords (numpy.ndarray): The N-dimensional coordinates.

    Returns:
        int or float: The value at the specified coordinates.

    Examples:
        >>> img = np.array([[1, 2], [3, 4]])
        >>> nd_coords = np.array([[0, 1], [1, 0]])
        >>> result = get_value_from_nd_coords(img, nd_coords)
        >>> print(result)
        [2, 3]

    """
    return img[convert_nd_coords_to_numpy_format(nd_coords)]

def convert_nd_coords_to_numpy_format(nd_coords):
    """
    Converts N-dimensional coordinates to the numpy format (tuple).

    Args:
        nd_coords (numpy.ndarray): The N-dimensional coordinates.

    Returns:
        tuple: The N-dimensional coordinates in the numpy format.

    Examples:
        >>> nd_coords = np.array([[1, 2, 3], [4, 5, 6]])
        >>> result = convert_nd_coords_to_numpy_format(nd_coords)
        >>> print(result)
        ((1, 4), (2, 5), (3, 6))

    """
    return tuple(nd_coords.T)

def set_coords_to_value(img, nd_coords, value):
    """
    Sets the value at the specified N-dimensional coordinates in the image.

    Args:
        img (numpy.ndarray): The input image.
        nd_coords (numpy.ndarray): The N-dimensional coordinates.
        value (int or float): The value to set.

    Examples:
        >>> img = np.zeros((3, 3))
        >>> nd_coords = np.array([[0, 0], [1, 1]])
        >>> value = 1
        >>> set_coords_to_value(img, nd_coords, value)
        >>> print(img)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 0.]]

    """
    img[convert_nd_coords_to_numpy_format(nd_coords)] = value

def convert_numpy_bbox_to_coord_pairs(bbox):
    """
    Converts a linear bounding box to coordinate pairs.

    Args:
        bbox (numpy.ndarray or list): The bounding box.

    Returns:
        numpy.ndarray: The coordinate pairs.

    Examples:
        >>> bbox = [1, 2, 3, 4, 5, 6]
        >>> result = convert_numpy_bbox_to_coord_pairs(bbox)
        >>> print(result)
        [[1 4]
         [2 5]
         [3 6]]

    """
    if not isinstance(bbox, np.ndarray):
        bbox = np.asarray(bbox)
    return np.reshape(bbox, (2, int(bbox.size / 2))).T


def get_image_bounds(original_image):
    """
    Returns the coordinate bounds of an image.

    Args:
        original_image (numpy.ndarray): The original image.

    Returns:
        numpy.ndarray: The coordinate bounds.

    Examples:
        >>> original_image = np.zeros((5, 5))
        >>> result = get_image_bounds(original_image)
        >>> print(result)
        [[0 5]
         [0 5]]

    """
    bounds = []
    for dim in range(len(original_image.shape)):
        bounds.append(0)
        bounds.append(original_image.shape[dim])

    bounds = np.asarray(bounds).reshape(-1, 2)
    return bounds


def filter_nan_rows(coords):
    """
    Filters out rows containing NaN values from a coordinate array.

    Args:
        coords (numpy.ndarray): The coordinate array.

    Returns:
        numpy.ndarray: The filtered coordinate array.

    Examples:
        >>> coords = np.array([[1, 2], [3, np.nan], [4, 5]])
        >>> result = filter_nan_rows(coords)
        >>> print(result)
        [[1. 2.]
         [4. 5.]]

    """
    if coords.size == 0:
        print('empty array --> returning array')
        return coords
    return coords[~np.isnan(coords).any(axis=1)]

def get_histogram_edges(min, max, increment):
    """
    Returns the bin edges for a histogram.

    Args:
        min (int or float): The minimum value.
        max (int or float): The maximum value.
        increment (int or float): The increment between bin edges.

    Returns:
        numpy.ndarray: The bin edges.

    Examples:
        >>> min_val = 0
        >>> max_val = 10
        >>> increment = 2
        >>> result = get_histogram_edges(min_val, max_val, increment)
        >>> print(result)
        [ 0  2  4  6  8 10]

    """
    bin_edges = np.arange(min, max + 1, increment)
    return bin_edges

def find_factors(arr):
    """
    Returns the unique values in an array.

    Args:
        arr (array-like): The input array.

    Returns:
        numpy.ndarray: An array containing the unique values in the input array.

    Examples:
        >>> find_factors(['apple', 'banana', 'banana', 'cherry', 'cherry', 'durian'])
        array(['apple', 'banana', 'cherry', 'durian'], dtype='<U6')

        >>> find_factors(['apple', 'apple', 'apple', 'apple'])
        array(['apple'], dtype='<U5')

        >>> find_factors(['cat', 'dog', 'elephant'])
        array(['cat', 'dog', 'elephant'], dtype='<U8')

    """
    unique_vals = np.unique(arr)
    return unique_vals

# def get_all_colors(arr):
#     return np.unique(arr)


def create_image_with_lines(image_size, x_spacing, y_spacing):
    """
    Create a black numpy image with lines every x_spacing pixels in the x direction
    and every y_spacing pixels in the y direction.

    Parameters:
    image_size (tuple): The size of the image (width, height).
    x_spacing (int): The spacing between vertical lines.
    y_spacing (int): The spacing between horizontal lines.

    Returns:
    numpy.ndarray: The image.
    """
    from skimage import draw, color
    # Create a black numpy image
    image = np.zeros(image_size, dtype=np.uint8)

    # Draw vertical lines every x_spacing pixels
    for x in range(0, image_size[0], x_spacing):
        print(x, 0, x, image_size[1])

        rr, cc = draw.line(x, 0, x, image_size[1]-1)
        image[rr, cc] = 255

    # Draw horizontal lines every y_spacing pixels
    for y in range(0, image_size[1], y_spacing):
        rr, cc = draw.line(0, y, image_size[0]-1, y)
        image[rr, cc] = 255

    return image

if __name__ == '__main__':
    if True:
        # Example usage
        image_size = (800, 600)  # Width, Height
        x_spacing = 50
        y_spacing = 100
        image = create_image_with_lines(image_size, x_spacing, y_spacing) # this grid image can be used to debug image crop --> TODO

        # Display the image
        plt.imshow(image)
        plt.show()

        sys.exit(0)


    print(get_histogram_edges(0,100,25)) # increment of 25 between 0 and 100 -−> give all the pairs of bounds for an histogram --> [  0  25  50  75 100]
    print(get_histogram_edges(0,100,10)) # increment of 10 between 0 and 100 -−> give all the pairs of bounds for an histogram --> [  0  10  20  30  40  50  60  70  80  90 100]



    test = np.zeros(shape=(32,32))
    coords = np.asarray([[0,0],[16,16],[20,20]])
    set_coords_to_value(test, coords, 255) # --> same as test[coords[:,0],coords[:,1]] # and what is great is that infinite nb of dims are supported --> can write generic code for 2D and 3D --> really cool --> replace everywhere in my code
    plt.imshow(test)
    plt.show()

