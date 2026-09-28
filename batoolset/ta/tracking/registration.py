from skimage.registration import phase_cross_correlation # change in scipy 0.19
# image = data.camera()
from skimage import transform
import numpy as np

__DEBUG__ = False

def apply_translation(img, y, x):
    """
    Applies translation to an image.

    Args:
        img (numpy.ndarray): The input image.
        y (float): Translation along the y-axis.
        x (float): Translation along the x-axis.

    Returns:
        numpy.ndarray: The translated image.
    """
    afine_tf = transform.AffineTransform(translation=(x, y))
    translated = transform.warp(img, inverse_map=afine_tf, order=0, preserve_range=True)
    return translated

def pre_register_images(orig_t0, orig_t1, apply_to_orig_t0=False):
    """
    Performs pre-registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        apply_to_orig_t0 (bool, optional): Whether to apply the translation to orig_t0. Defaults to False.

    Returns:
        tuple or numpy.ndarray: If apply_to_orig_t0 is False, returns the global translations (gloabl_t0, gloabl_t1).
        If apply_to_orig_t0 is True, returns the translated orig_t0 image.

    Notes:
        - The function uses phase cross-correlation to estimate the global translation between the two images.
        - The global translation is then applied to either orig_t0 or both orig_t0 and orig_t1, depending on the
          value of apply_to_orig_t0.
    """
    global_translation, error, diffphase = phase_cross_correlation(orig_t0, orig_t1)
    gloabl_t0 = -global_translation[0]
    gloabl_t1 = -global_translation[1]
    if not apply_to_orig_t0:
        return gloabl_t0, gloabl_t1
    else:
        return apply_translation(orig_t0, -gloabl_t0, -gloabl_t1)

def _translated_tile(img, y, x, tile_height, tile_width, ty, tx):
    """
    Returns apply_translation(img, ty, tx)[y:y + tile_height, x:x + tile_width] (0 outside of the image)
    without warping the whole image. Only valid for integer translations of 2D images (nearest neighbour warp of an
    integer translation is a plain shift).
    """
    height, width = img.shape[:2]
    y_end = min(y + tile_height, height)
    x_end = min(x + tile_width, width)
    # same output dtype as transform.warp (depends on the skimage version for order=0)
    warp_dtype = transform.warp(img[:1, :1], inverse_map=transform.AffineTransform(), order=0, preserve_range=True).dtype
    out = np.zeros((y_end - y, x_end - x), dtype=warp_dtype)
    ty = int(ty)
    tx = int(tx)
    # out[r, c] = img[r - ty, c - tx] for the source pixels inside the image
    r0 = max(y, ty)
    r1 = min(y_end, height + ty)
    c0 = max(x, tx)
    c1 = min(x_end, width + tx)
    if r0 < r1 and c0 < c1:
        out[r0 - y:r1 - y, c0 - x:c1 - x] = img[r0 - ty:r1 - ty, c0 - tx:c1 - tx]
    return out


def get_pyramidal_registration(orig_t0, orig_t1, depth=2, threshold_translation=20, halve_threshold=True,
                               min_valid_fraction=0.75):
    """
    Performs pyramidal registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        depth (int, optional): The depth of the pyramid. Defaults to 2.
        threshold_translation (float, optional): The threshold for ignoring large translations. Defaults to 20.
        halve_threshold (bool, optional): the largest correction accepted at pyramid level i is
            threshold_translation / 2 ** (i + 1) instead of threshold_translation. Small tiles of an epithelium hold
            a few similar cells, their phase correlation often peaks one cell away: with 20 px at every level, 15 % of
            the cells of a real movie were registered more than one cell radius away (vs 0.6 % with this option),
            which made the dynamic tracker slide IDs by one cell over whole regions.
        min_valid_fraction (float, optional): tiles whose translated source is less than this fraction inside the
            image are not refined (the zeros outside biased the correlation).

    Returns:
        numpy.ndarray: The translation matrix representing the translations for each pixel.

    Notes:
        - The function applies pre-registration between orig_t0 and orig_t1.
        - It then iteratively performs registration on blocks of decreasing size.
        - Translations are accumulated in the translation matrix.
    """

    translation_matrix = np.zeros((*orig_t0.shape, 2))

    gloabl_t0, gloabl_t1 = pre_register_images(orig_t0, orig_t1)

    translation_matrix[..., 0] += gloabl_t0
    translation_matrix[..., 1] += gloabl_t1

    height = orig_t0.shape[0]
    width = orig_t0.shape[1]

    block_size = 256

    for i in range(depth):
        try:
            tile_height = block_size // (i * 2)
            tile_width = block_size // (i * 2)
        except:
            tile_height = block_size
            tile_width = block_size

        threshold = threshold_translation / 2 ** (i + 1) if halve_threshold else threshold_translation

        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                ty = translation_matrix[y, x, 0]
                tx = translation_matrix[y, x, 1]
                if min_valid_fraction > 0:
                    # fraction of the tile whose source pixel (p - t) is inside the image
                    y_end = min(y + tile_height, height)
                    x_end = min(x + tile_width, width)
                    rows = max(0, min(y_end, height + ty) - max(y, ty))
                    cols = max(0, min(x_end, width + tx) - max(x, tx))
                    if rows * cols < min_valid_fraction * (y_end - y) * (x_end - x):
                        continue
                if orig_t0.ndim == 2 and float(ty).is_integer() and float(tx).is_integer():
                    # phase_cross_correlation shifts are integers --> exact shifted copy of the tile only
                    tile_t0 = _translated_tile(orig_t0, y, x, tile_height, tile_width, ty, tx)
                else:
                    afine_tf = transform.AffineTransform(
                        translation=(-translation_matrix[y, x, 1], -translation_matrix[y, x, 0]))
                    translated = transform.warp(orig_t0, inverse_map=afine_tf, order=0, preserve_range=True)
                    tile_t0 = translated[y:y + tile_height, x:x + tile_width]
                tile_t1 = orig_t1[y:y + tile_height, x:x + tile_width]

                shift, error, diffphase = phase_cross_correlation(tile_t0, tile_t1)
                t0 = -shift[0]
                t1 = -shift[1]

                if abs(t0) > threshold or abs(t1) > threshold:
                    continue

                translation_matrix[y:y + tile_height, x:x + tile_width, 0] += t0
                translation_matrix[y:y + tile_height, x:x + tile_width, 1] += t1

    return translation_matrix
