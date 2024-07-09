# this will contain basic tools to do unit conversions from one dimension to another, e.g. cm to inches
# https://pixelcalculator.com/fr -> cool to test

def cm_to_inch(size_in_cm):
    return size_in_cm / 2.54


def scaling_factor_to_achieve_DPI(desired_dpi, dpi=72):
    return desired_dpi / dpi


def inch_to_cm(size_in_inch):
    """
    Convert a length from inches to centimeters.

    Args:
        size_in_inch (float): The length in inches.

    Returns:
        float: The length in centimeters.
    """
    return size_in_inch * 2.54


def pixels_to_cm(pixels, dpi=72):
    """
    Convert a length from pixels to centimeters, given a DPI resolution.

    Args:
        pixels (int): The length in pixels.
        dpi (float): The desired DPI (dots per inch) resolution.

    Returns:
        float: The length in centimeters.
    """
    inches = pixels / dpi
    return inches * 2.54


def pixels_to_inch(pixels, dpi=72):
    """
    Convert a length from pixels to inches, given a DPI resolution.

    Args:
        pixels (int): The length in pixels.
        dpi (float): The desired DPI (dots per inch) resolution.

    Returns:
        float: The length in inches.
    """
    return pixels / dpi

def cm_to_pixels(cm, dpi=72):
    """
    Convert a length from centimeters to pixels, given a DPI resolution.

    Args:
        cm (float): The length in centimeters.
        dpi (float): The desired DPI (dots per inch) resolution.

    Returns:
        int: The length in pixels.
    """
    inches = cm / 2.54
    return int(inches * dpi)


def inch_to_pixels(inch, dpi=72):
    """
    Convert a length from inches to pixels, given a DPI resolution.

    Args:
        inch (float): The length in inches.
        dpi (float): The desired DPI (dots per inch) resolution.

    Returns:
        int: The length in pixels.
    """
    return int(inch * dpi)


if __name__ == '__main__':
    print(cm_to_pixels(21))