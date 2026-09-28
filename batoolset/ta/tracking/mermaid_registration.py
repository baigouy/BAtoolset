"""
Optional non-rigid registration of consecutive frames with mermaid (https://github.com/uncbiag/mermaid, PyTorch, GPU
strongly advised): gives, for every pixel of frame t, where it goes in frame t+1. The trackers use it instead of their
own registration when the maps have been computed (compute_mermaid_maps):
    - dynamic tissue tracker: as its translation_matrix (t0 cell centroid + displacement -> label at t1)
    - static tracker (match by max overlap): the t0 track image is warped onto t1 before matching

Output per frame t (except the last): <TA folder of t>/mermaid_translation_matrix.tif, H x W x 2 float32, same format
as the dynamic tracker's translation_matrix.tif: pixel (y, x) of t moves to (y + [..., 0], x + [..., 1]) at t+1. The
global pre-registration is included, so the map is self-contained.

Mermaid is not a dependency: it is imported from the Python path, from $MERMAID_PATH or from a 'mermaid' folder (the
repository root, holding the 'mermaid' package) next to BAtools_pkg. Its old code needed a few fixes for PyTorch >= 1.8
(torch.rfft/irfft, ReduceLROnPlateau(verbose)).

Settings (validated on 512 x 512 pupal notum frames, RTX 5090): scalar-momentum SVF model, NCC similarity, multi-Gaussian
smoother, 5 time steps, map at half resolution, 100 SGD iterations: ~5-15 s per pair (20 time steps at full resolution:
40-120 s, same accuracy). On 3-frame gaps it gave 8 and 7 wrongly predicted cells out of ~550 vs 18 and 10 for the
pyramidal translation; on consecutive frames of that movie (motion of a few pixels) it is not better than a translation.
"""
import contextlib
import io
import os
import sys
import numpy as np
from batoolset.img import Img
from batoolset.files.tools import get_TA_file
from batoolset.ta.tracking.registration import pre_register_images
from batoolset.tools.early_stopper_class import early_stop

MAP_NAME = 'mermaid_translation_matrix.tif'

# mermaid parameters (the model name set here is the one used: a model given in a json file overrides model_name)
MERMAID_PARAMETERS = {
    'model': {
        'deformation': {'compute_similarity_measure_at_low_res': False, 'map_low_res_factor': 0.5, 'use_map': True},
        'registration_model': {
            'env': {'get_momentum_from_external_network': False, 'reg_factor': 1.0, 'use_ode_tuple': False,
                    'use_odeint': True},
            'forward_model': {
                'adjoin_on': True, 'atol': 1e-05, 'rtol': 1e-05, 'number_of_time_steps': 5, 'solver': 'rk4',
                'tFrom': 0.0, 'tTo': 1.0,
                'smoother': {'type': 'multiGaussian', 'multi_gaussian_stds': [0.05, 0.1, 0.15, 0.2, 0.25],
                             'multi_gaussian_weights': [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]}},
            'forward_modelF': {'tTo': 1.0},
            'loss': {'display_max_displacement': False, 'limit_displacement': False, 'max_displacement': 0.05},
            'similarity_measure': {'develop_mod_on': False, 'sigma': 0.1, 'type': 'ncc'},
            'spline_order': 1,
            'type': 'svf_scalar_momentum_map',
            'use_CFL_clamping': True}},
    'optimizer': {
        'gradient_clipping': {'clip_display': False, 'clip_individual_gradient': True,
                              'clip_individual_gradient_value': 1.0, 'clip_shared_gradient': True,
                              'clip_shared_gradient_value': 1.0},
        'name': 'sgd',
        'scheduler': {'factor': 0.5, 'patience': 10, 'verbose': False},
        'sgd': {'individual': {'dampening': 0.0, 'lr': 0.05, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.0},
                'shared': {'dampening': 0.0, 'lr': 0.05, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.0}},
        'single_scale': {'nr_of_iterations': 100, 'rel_ftol': 1e-07},
        'use_step_size_scheduler': True,
        'weight_clipping_type': 'none',
        'weight_clipping_value': 1.0}}


def _mermaid_folders():
    folders = []
    if os.environ.get('MERMAID_PATH'):
        folders.append(os.environ['MERMAID_PATH'])
    # BAtools_pkg/batoolset/ta/tracking/this_file -> folder containing BAtools_pkg
    here = os.path.dirname(os.path.abspath(__file__))
    folders.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here)))), 'mermaid'))
    return folders


def load_mermaid():
    """
    Imports mermaid (see module doc for where it is searched).

    Returns:
        module or None: the mermaid package, None if it (or PyTorch) is not available.
    """
    try:
        import mermaid
        if getattr(mermaid, '__file__', None) is not None:
            return mermaid
        # a folder named mermaid (the repository root) shadowed the package
        del sys.modules['mermaid']
    except ImportError:
        pass
    for folder in _mermaid_folders():
        if os.path.isfile(os.path.join(folder, 'mermaid', '__init__.py')):
            sys.path.insert(0, folder)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    import mermaid
                if getattr(mermaid, '__file__', None) is not None:
                    return mermaid
            except ImportError:
                pass
            sys.path.remove(folder)
            sys.modules.pop('mermaid', None)
    return None


def is_mermaid_available():
    return load_mermaid() is not None


def _normalize(img):
    img = np.asarray(img, dtype=np.float32)
    span = img.max() - img.min()
    if span == 0:
        return np.zeros_like(img)
    return (img - img.min()) / span


def _shifted(img, ty, tx):
    """img(p + t) (nearest pixel, edges repeated)"""
    rows, cols = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.clip(rows + int(round(ty)), 0, img.shape[0] - 1)
    c = np.clip(cols + int(round(tx)), 0, img.shape[1] - 1)
    return img[r, c]


def mermaid_translation_matrix(img_t0, img_t1, nr_of_iterations=None, verbose=False):
    """
    Non-rigid registration of two 2D images.

    Args:
        img_t0 (numpy.ndarray): image at t (2D).
        img_t1 (numpy.ndarray): image at t+1 (2D, same shape).
        nr_of_iterations (int, optional): optimizer iterations (default MERMAID_PARAMETERS, 100).
        verbose (bool): show mermaid's output.

    Returns:
        numpy.ndarray: H x W x 2 float32, pixel (y, x) of img_t0 is at (y, x) + [y, x, :] in img_t1.
    """
    load_mermaid()
    import torch
    import mermaid.module_parameters as pars
    from mermaid.simple_interface import RegisterImagePair

    I0 = _normalize(img_t0)
    I1 = _normalize(img_t1)
    # global translation first (mermaid does not like large displacements): t1 ~ t0 moved by (ty, tx)
    ty, tx = pre_register_images(I0, I1)
    ty = int(round(ty))
    tx = int(round(tx))
    I1_back = np.ascontiguousarray(_shifted(I1, ty, tx))
    spacing = np.asarray([1. / I0.shape[0], 1. / I0.shape[1]])
    params_dict = MERMAID_PARAMETERS if nr_of_iterations is None else _with_iterations(nr_of_iterations)
    out = None if verbose else io.StringIO()
    with contextlib.redirect_stdout(out) if out is not None else contextlib.nullcontext():
        params = pars.ParameterDict(printSettings=verbose)
        # same as load_JSON (ParameterDict cannot be built from a dict)
        params.ext = _copy(params_dict)
        si = RegisterImagePair()
        # source = t1 (moved back), target = t0: the map gives, for every t0 pixel, its position in t1
        si.register_images(I1_back[np.newaxis, np.newaxis], I0[np.newaxis, np.newaxis], spacing,
                           model_name=params_dict['model']['registration_model']['type'], visualize_step=None,
                           use_multi_scale=False, params=params, recording_step=None)
        phi = si.get_map().detach().cpu().numpy()[0]
    del si
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows, cols = np.mgrid[0:I0.shape[0], 0:I0.shape[1]].astype(np.float32)
    translation_matrix = np.zeros((*I0.shape, 2), dtype=np.float32)
    translation_matrix[..., 0] = phi[0] / spacing[0] + ty - rows
    translation_matrix[..., 1] = phi[1] / spacing[1] + tx - cols
    return translation_matrix


def _copy(d):
    import copy
    return copy.deepcopy(d)


def _with_iterations(nr_of_iterations):
    params = _copy(MERMAID_PARAMETERS)
    params['optimizer']['single_scale']['nr_of_iterations'] = int(nr_of_iterations)
    return params


def _channel(img, channel):
    if img.ndim > 2:
        if channel is None:
            return img.mean(axis=-1)
        return img[..., channel]
    return img


def compute_mermaid_maps(lst, channel=None, overwrite=False, nr_of_iterations=None, progress_callback=None):
    """
    Computes and saves the mermaid map of every pair of consecutive frames (see module doc).

    Args:
        lst (list): the images of the movie, in time order.
        channel (int, optional): channel to register (membranes); None = mean of the channels.
        overwrite (bool): recompute existing maps.
        nr_of_iterations (int, optional): optimizer iterations.
        progress_callback (optional): Qt progress signal (emit(int)).

    Returns:
        list: the files written.
    """
    if load_mermaid() is None:
        raise ImportError('mermaid is not available: put the mermaid repository next to BAtools_pkg or set MERMAID_PATH')
    written = []
    for i, (file_t0, file_t1) in enumerate(zip(lst, lst[1:])):
        try:
            if early_stop.stop:
                return written
        except Exception:
            pass
        if progress_callback is not None:
            progress_callback.emit(int(i * 100 / max(len(lst) - 1, 1)))
        else:
            print('mermaid registration %d/%d' % (i + 1, len(lst) - 1))
        output = get_TA_file(os.path.splitext(file_t0)[0], MAP_NAME)
        if os.path.isfile(output) and not overwrite:
            continue
        img_t0 = _channel(np.asarray(Img(file_t0)), channel)
        img_t1 = _channel(np.asarray(Img(file_t1)), channel)
        translation_matrix = mermaid_translation_matrix(img_t0, img_t1, nr_of_iterations=nr_of_iterations)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        Img(translation_matrix, dimensions='hwc').save(output)
        written.append(output)
    return written


def load_mermaid_translation_matrix(filename_without_ext, shape=None):
    """
    Returns:
        numpy.ndarray or None: the saved map of this frame (to the next one), None if missing or of another shape.
    """
    path = get_TA_file(filename_without_ext, MAP_NAME)
    if not os.path.isfile(path):
        return None
    translation_matrix = np.asarray(Img(path), dtype=np.float32)
    if translation_matrix.ndim != 3 or translation_matrix.shape[-1] != 2:
        return None
    if shape is not None and tuple(translation_matrix.shape[:2]) != tuple(shape[:2]):
        return None
    return translation_matrix


def warp_with_translation_matrix(image_t0, translation_matrix):
    """
    Moves image_t0 (e.g. a track ID image) onto frame t+1, nearest pixel (IDs are not blended). The value at a t+1
    pixel p is taken at p - displacement(p) in t0 (the displacement field is smooth, so this is its inverse up to
    second order terms).
    """
    height, width = image_t0.shape[:2]
    rows, cols = np.mgrid[0:height, 0:width]
    r = np.rint(rows - translation_matrix[..., 0]).astype(int)
    c = np.rint(cols - translation_matrix[..., 1]).astype(int)
    outside = (r < 0) | (r >= height) | (c < 0) | (c >= width)
    warped = image_t0[np.clip(r, 0, height - 1), np.clip(c, 0, width - 1)]
    # tissue coming from outside of t0: 0, as apply_translation does
    warped[outside] = 0
    return warped
