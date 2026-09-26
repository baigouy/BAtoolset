'''ImageReader: low-level file-format reading for batoolset.

Split out of img.py (2026-07-23) since it was one of the largest, most
self-contained chunks of that file. ImageReader never constructs an Img
itself -- it only returns (metadata, ndarray) tuples -- so this module has no
top-level dependency on batoolset.img. The one exception is imageread()'s use
of to_stack() (which does construct Img objects internally), and read()'s use
of a few small helpers (VIDEO_EXTS, is_video_file, read_video,
read_file_from_url); those are imported locally inside the methods that need
them rather than at module level, so this module and batoolset.img can be
imported in either order without a circular-import error.

Img.__new__ imports ImageReader from here and batoolset.img re-exports it, so
existing code doing `from batoolset.img import ImageReader` is unaffected.
'''

import json
import traceback

import numpy as np
import skimage
from skimage import io  # noqa: F401 -- registers skimage.io as an attribute of the skimage package
import tifffile  # open Zeiss .tif and .lsm files
import read_lif  # read Leica .lif files (requires numexpr)

from batoolset.tools.logger import TA_logger  # logging
logger = TA_logger()  # logging_level=TA_logger.DEBUG


class ImageReader:

    def read(f, serie_to_open=None):

        # deferred import: to_stack (transitively, via imageread()) constructs Img,
        # and Img.__new__ imports ImageReader from this module, so these must stay
        # function-local to avoid a circular import regardless of which module a
        # caller happens to import first.
        from batoolset.img import VIDEO_EXTS, is_video_file, read_video, read_file_from_url

        width = None
        height = None
        depth = None
        channels = None
        voxel_x = None
        voxel_y = None
        voxel_z = None
        times = None
        bits = None
        t_frames = None
        luts = None
        ar = None
        overlays = None
        roi = None
        timelapse = None
        creation_time = None

        dimensions_string = ''

        metadata = {'w': width, 'h': height, 'c': channels, 'd': depth, 't': t_frames, 'bits': bits, 'vx': voxel_x,
                    'vy': voxel_y, 'vz': voxel_z, 'AR': ar, 'dimensions': dimensions_string, 'LUTs': luts,
                    'times': times, 'Overlays': overlays, 'ROI': roi, 'timelapse': timelapse,
                    'creation_time': creation_time}  # TODO check always ok

        logger.debug('loading' + str(f))

        # videos are decoded natively as time-stacks (xy-t) — handle before anything else
        if is_video_file(f):
            return read_video(f)

        # by default we now open with imageJ
        _IJ_EXCLUDED_EXTS = ('.tif', '.tiff', '.xls','.xlsx','.lst','.db','.txt','.png', '.jpg', '.jpeg', '.bmp', '.zarr') + VIDEO_EXTS  # formats handled natively
        _ij_reader_available = False
        try:
            from batoolset.IJ.ij_reader import list_and_open_series, get_luts, to_metadata
            import imagej as _imagej
            _ij_reader_available = True
        except ImportError:
            logger.info('IJ loader not found...')
            pass

        if _ij_reader_available and isinstance(f, str) and not f.lower().endswith(_IJ_EXCLUDED_EXTS):
            try:
                # print('opening image with IJ')
                # lazy-init ij gateway (only once per process)
                import batoolset.IJ.ij_reader as _ij_reader
                if getattr(_ij_reader, '_ij_gateway', None) is False:
                    raise RuntimeError('Fiji init previously failed; skipping')
                if not hasattr(_ij_reader, '_ij_gateway') or _ij_reader._ij_gateway is None:
                    try:
                        _ij_reader._ij_gateway = _ij_reader.init_ij()
                    except Exception:
                        _ij_reader._ij_gateway = False  # prevent future retries
                        raise
                _ij = _ij_reader._ij_gateway

                imp = list_and_open_series(f, series_index=serie_to_open if serie_to_open is not None else 0)

                if imp is not None:
                    luts = get_luts(imp)
                    meta = to_metadata(imp, luts=luts, filepath=f,
                                       series_index=serie_to_open if serie_to_open is not None else 0)

                    # convert numpy array from imp
                    w_ij = imp.getWidth()
                    h_ij = imp.getHeight()
                    n_c  = imp.getNChannels()
                    n_z  = imp.getNSlices()
                    n_t  = imp.getNFrames()
                    try:
                        image = _ij.py.from_java(imp)
                        image = np.array(image)
                        # from_java may give channels-first (e.g. (C,H,W)); detect and
                        # move channels to last axis so layout always matches dim_order.
                        if n_c > 1 and image.ndim >= 3:
                            # Find which axis has size == n_c and is not H or W
                            # (H=h_ij, W=w_ij are always the last two in tdhwc order)
                            for ax in range(image.ndim - 2):
                                if image.shape[ax] == n_c:
                                    image = np.moveaxis(image, ax, -1)
                                    break
                    except AttributeError:
                        # Fallback: ij.py.from_java can fail in headless/interactive mode.
                        # Build the array slice by slice and stack channels on the LAST
                        # axis so the result is always in TDHWC order.
                        bits = imp.getBitDepth()
                        dtype = np.uint8 if bits == 8 else (np.float32 if bits == 32 else np.uint16)
                        stack = imp.getStack()
                        frames = []  # one entry per (t, z) — shape (H, W) or (H, W, C)
                        for t in range(1, n_t + 1):
                            for z in range(1, n_z + 1):
                                ch_slices = []
                                for c in range(1, n_c + 1):
                                    idx = imp.getStackIndex(c, z, t)
                                    arr = np.array(stack.getProcessor(idx).getPixels(), dtype=dtype)
                                    ch_slices.append(arr.reshape(h_ij, w_ij))
                                if n_c > 1:
                                    frames.append(np.stack(ch_slices, axis=-1))  # (H, W, C)
                                else:
                                    frames.append(ch_slices[0])                  # (H, W)
                        if len(frames) == 1:
                            image = frames[0]
                        else:
                            image = np.stack(frames)  # (T*D, H, W) or (T*D, H, W, C)

                    # Build dimension string in TDHWC order (only present axes, C last)
                    dim_order_str = ''
                    if n_t > 1: dim_order_str += 't'
                    if n_z > 1: dim_order_str += 'd'
                    dim_order_str += 'hw'
                    if n_c > 1: dim_order_str += 'c'

                    # map to_metadata keys → ImageReader metadata keys
                    metadata.update({
                        'w': w_ij,
                        'h': h_ij,
                        'd': n_z if n_z > 1 else None,
                        'c': n_c if n_c > 1 else None,
                        't': n_t if n_t > 1 else None,
                        'bits': meta['bits'],
                        'vx': meta['vx'],
                        'vy': meta['vy'],
                        'vz': meta['vz'],
                        'AR': meta['AR'],
                        'LUTs': luts,
                        'dimensions': dim_order_str,
                        'Overlays': meta['Overlays'] if meta['Overlays'] else None,
                        'ROI': meta['ROI'],
                        'timelapse': meta['timelapse'],
                        'creation_time': meta['creation_time'],
                    })
                    logger.debug('ij_reader successfully opened ' + str(f))
                    return metadata, image
            except Exception:
                traceback.print_exc()
                logger.warning('ij_reader failed for ' + str(f) + ' → falling back to default reader')

        # small hack to allow for opening web hosted images (e.g. from https://samples.fiji.sc/)
        if isinstance(f, str):
            if f.lower().startswith('http') or f.lower().startswith('file:'):
                to_read = read_file_from_url(f)
            else:
                to_read = f
        else:
            to_read = f

        # TODO skip metadata loading if I just wanna have the image or can I fuse this one with the next ???
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff') or f.lower().endswith(
                '.lsm'):

            # added support for url
            with tifffile.TiffFile(to_read) as tif:

                # TODO need handle ROIs there!!!
                # just copy stuff
                # --> can then use it and pass it directly then if needed --> maybe need a smart handling in case there is a reduction of the number of dimensions to only keep the correct ROIs

                # if image is IJ image preserve ROIs and overlays
                if tif.is_imagej:
                    if 'Overlays' in tif.imagej_metadata:
                        overlays = tif.imagej_metadata['Overlays']
                        metadata['Overlays'] = overlays
                    if 'ROI' in tif.imagej_metadata:
                        roi = tif.imagej_metadata['ROI']
                        metadata['ROI'] = roi
                    if 'LUTs' in tif.imagej_metadata:
                        luts = tif.imagej_metadata['LUTs']
                        metadata['LUTs'] = luts

                tif_tags = {}
                for tag in tif.pages[0].tags.values():

                    # if name == 'vx':
                    #     width = value
                    # elif name == 'vy':
                    #     height = value
                    # elif name == 'vz':
                    #     depth = value
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                    logger.debug(''' + name + ''' + '\'' + str(value) + '\'')
                    if name == 'ImageWidth':
                        width = value
                    elif name == 'ImageLength':
                        height = value
                    elif name == 'BitsPerSample':
                        if not isinstance(value, tuple):
                            bits = value
                        else:
                            bits = value[0]
                    elif name == 'XResolution':
                        val = value[1] / value[0]
                        if val != 1.0 or voxel_x is None:
                            voxel_x = value[1] / value[0]
                            if voxel_x == 1./72.: # most likely a dpi crap 1/72 that is useless
                                voxel_x=None
                        # print('grabouypo', value[1] / value[0], value[1], value[0])
                    elif name == 'YResolution':
                        val = value[1] / value[0]
                        if val != 1.0 or voxel_x is None:
                            voxel_y = val
                            if voxel_y == 1./72.:# most likely a dpi crap 1/72 that is useless
                                voxel_y=None
                    elif name == 'ImageDescription':
                        try:
                        # print('value azeaze', value)
                            if not value.startswith('{'):
                                # print(value)
                                lines = value.split()
                                # print(len(lines), lines)
                                # print(type(value), value)
                                #
                                # if value.startswith('{'):
                                #     res = json.loads(value)
                                #     print(res)
                                #     print(res['vx']+1)
                                # then it is a dict and I need parse it back
                                # if description starts with a {
                                for l in lines:
                                    logger.debug('1'' + l + ''1')
                                    if l.startswith('channels'):
                                        _, val = l.split('=')
                                        channels = int(val)
                                    elif l.startswith('slices'):  # Z slices
                                        _, val = l.split('=')
                                        depth = int(val)
                                    elif l.startswith('frames'):  # time frames
                                        _, val = l.split('=')
                                        t_frames = int(val)
                                    elif l.startswith('spacing'):
                                        _, val = l.split('=')
                                        voxel_z = float(val)

                                        # print(name, value)
                                        # TODO if there is an ImageDescription I could parse it and get the data out of it
                                        # ImageDescription
                        except Exception:
                            # traceback.print_exc()
                            logger.error('error reading description'+str(value))

                        else:
                            try:
                                # TODO improve that
                                # metadata added is in fact epyseg metadata --> recover it
                                epyseg_meta = json.loads(value)
                                if 'vx' in epyseg_meta:
                                    voxel_x = epyseg_meta['vx']
                                if 'vy' in epyseg_meta:
                                    voxel_y = epyseg_meta['vy']
                                if 'vz' in epyseg_meta:
                                    voxel_z = epyseg_meta['vz']
                                if 'creation_time' in epyseg_meta:
                                    creation_time = epyseg_meta['creation_time']
                                if 'timelapse' in epyseg_meta:
                                    timelapse = epyseg_meta['timelapse']
                                if 'times' in epyseg_meta:
                                    times = epyseg_meta['times']
                            except Exception:
                                logger.debug('metadata ignored', value)
                                # e.g. ('ImageJ=1.40g',) --> this is not read
                            # TODO maybe get more metadata from epyseg --> check which meta is smart to keep and which isn't...

                    # read lsm
                    if isinstance(value, dict):
                        for name, value in value.items():
                            logger.debug(name + ' ' + str(value))
                            # THE 3 BELOW ARE MY OWN TAGS --> ADD MORE AND MAYBE STICK TO OMERO FOR CONSISTENCY MAYBE BUT OK FOR NOW
                            if name == 'DimensionZ':
                                depth = value
                            elif name == 'DimensionX':
                                width = value
                            elif name == 'DimensionY':
                                height = value
                            elif name == 'DimensionTime':
                                t_frames = value
                                if t_frames == 1:
                                    t_frames = None
                            elif name == 'DimensionChannels':
                                channels = value
                            elif name == 'VoxelSizeX':
                                voxel_x = value * 1_000_000
                            elif name == 'VoxelSizeY':
                                voxel_y = value * 1_000_000
                            elif name == 'VoxelSizeZ':
                                voxel_z = value * 1_000_000
                            elif name == 'TimeStamps':
                                times = value
                            elif name == 'ChannelColors':
                                luts = value['Colors']
                # print('in here',tif.series[0].asarray()) # --> so indeed it can read it here --> the rest is then a waste of time
                # tif.series[0].asarray() # just one page
                # image = np.squeeze(np.asarray(tif.series)) # not good because reads only one page here too --> at some point work on that
        # TODO also recover my own tags maybe # TODO recode all of this properly some day because it becomes a huge mess now

        # very dumb thing here is that I open the tiff file twice --> really not Smart --> change this --> I opne it here and above --> indeed
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff') or f.lower().endswith('.lsm'):
            if not isinstance(to_read, str):
                # assume url file
                to_read.seek(0)

            # image_stack = tifffile.imread(to_read)
            # # has more properties than that
            # image = image_stack
            # image = np.squeeze(image)
            image = np.squeeze(tifffile.imread(to_read))
        elif f.lower().endswith('.lif'):
            # reader = read_lif.Reader(f)
            # series = reader.getSeries()
            # # print('series', len(series))
            # chosen = series[0]
            #
            # meta_data = chosen.getMetadata()
            # voxel_x = meta_data['voxel_size_x']
            # voxel_y = meta_data['voxel_size_y']
            # voxel_z = meta_data['voxel_size_z']
            # width = meta_data['voxel_number_x']
            # height = meta_data['voxel_number_y']
            # depth = meta_data['voxel_number_z']
            # channels = meta_data['channel_number']
            # times = chosen.getTimeStamps()
            # t_frames = chosen.getNbFrames()
            #
            # image = None
            # for i in range(channels):
            #     cur_image = chosen.getFrame(channel=i)
            #     dimName = {1: 'X',
            #                2: 'Y',
            #                3: 'Z',
            #                4: 'T',
            #                5: 'Lambda',
            #                6: 'Rotation',
            #                7: 'XT Slices',
            #                8: 'TSlices',
            #                10: 'unknown'}
            #     cur_image = np.moveaxis(cur_image, -1, 0)
            #     if image is None:
            #         image = cur_image
            #     else:
            #         image = np.stack((image, cur_image), axis=-1)
            image = None
            reader = read_lif.Reader(f)
            series = reader.getSeries() # shall I do smthg to open all ?? maybe as a list --> can be useful for projections and alike
            # print('series', len(series))
            if serie_to_open is None:
                chosen = series[0]
            else:
                if serie_to_open >= len(series) or serie_to_open < 0:
                    logger.error('Out of range serie nb for current lif file, returning None')
                    return None
                chosen = series[serie_to_open]

            meta_data = chosen.getMetadata()
            voxel_x = meta_data['voxel_size_x']
            voxel_y = meta_data['voxel_size_y']
            voxel_z = meta_data['voxel_size_z']
            width = meta_data['voxel_number_x']
            height = meta_data['voxel_number_y']
            depth = meta_data['voxel_number_z']
            channels = meta_data['channel_number']
            times = chosen.getTimeStamps()  # shall I try getRelativeTimeStamps????
            # print('relative times', chosen.getRelativeTimeStamps()) # marche pas peut etre sur non chosen --> still useless for lif
            timelapse = chosen.getTimeLapse()
            t_frames = chosen.getNbFrames()

            # print('t_frames', t_frames)
            # TODO check time points cause I think they are not ok for the t frames

            # stack = None
            for T in range(t_frames):
                zstack = None
                for i in range(channels):
                    cur_image = chosen.getFrame(T=T, channel=i)
                    # dimName = {1: 'X',
                    #            2: 'Y',
                    #            3: 'Z',
                    #            4: 'T',
                    #            5: 'Lambda',
                    #            6: 'Rotation',
                    #            7: 'XT Slices',
                    #            8: 'TSlices',
                    #            10: 'unknown'}
                    cur_image = np.moveaxis(cur_image, -1, 0)
                    if zstack is None:
                        zstack = cur_image
                    else:
                        zstack = np.stack((zstack, cur_image), axis=-1)
                if image is None:
                    image = zstack[np.newaxis, ...]
                    # stack = image
                else:
                    # print(image.shape, zstack.shape)
                    image = np.vstack((image, zstack[np.newaxis, ...]))
                    # stack = np.vstack((stack, image), axis = np.newaxis)

            # if only one T --> reduce dimensionality
            if t_frames == 1:
                t_frames = None

            # print('before squeeze', image.shape)
            image = np.squeeze(image)
            # image = stack
        elif f.lower().endswith('.zarr'):
            try:
                import zarr as _zarr
            except ImportError:
                raise ImportError("zarr is required to read .zarr files. Install with: pip install zarr")
            _store = _zarr.open(f, mode='r')
            # prefer 'mask' dataset (epyseg convention); fall back to first key
            if 'mask' in _store:
                _arr = _store['mask']
            else:
                _keys = list(_store.keys())
                if not _keys:
                    return metadata, np.zeros((1, 1), dtype=np.uint8)
                _arr = _store[_keys[0]]
            image = np.asarray(_arr)
            # build dimension string from ndim
            _ndim = image.ndim
            if _ndim == 2:
                dimensions = 'hw'
                height, width = image.shape
            elif _ndim == 3:
                dimensions = 'dhw'
                depth, height, width = image.shape
            elif _ndim == 4:
                # heuristic: last axis ≤ 4 → channels, otherwise time
                if image.shape[-1] <= 4:
                    dimensions = 'dhwc'
                    depth, height, width, channels = image.shape
                else:
                    dimensions = 'tdhw'
                    t_frames, depth, height, width = image.shape
            else:
                dimensions = None
            bits = image.dtype.itemsize * 8
            metadata.update({
                'dimensions': dimensions,
                'bits': bits,
                'h': height,
                'w': width,
                'd': depth if _ndim >= 3 else None,
                't': t_frames if _ndim == 4 and dimensions == 'tdhw' else None,
                'c': channels if _ndim == 4 and dimensions == 'dhwc' else None,
            })
            return metadata, image
        else:
            if not f.lower().endswith('.npy') and not f.lower().endswith('.npz'):
                # for some reason this stuff reads 8 bits images as RGB and that causes some trouble
                # image = skimage.io.imread(f) # shall I replace by imageio ???
                # image = imageio.imread(f) # ok
                # image = imageio_plugin.imread(f) # not ok

                # import imageio
                # from skimage.io._plugins import imageio_plugin
                # too dangerous to change code for now
                # try:
                #     image = imageio.v2.imread(f)  # ok
                # except:
                    image = skimage.io.imread(f)
                    try:
                        if image.shape[0]==1: #the channels of /E/Sample_images/sample_images_FIJI/blobs.png are not read in the proper order which creates some errors !!!--> they are chw instead of hwc !!!
                            image = np.squeeze(image) # somehow there is a bug in the dimension order when reading the png /E/Sample_images/sample_images_FIJI/blobs.png --> need a fix

                            # image '/E/Sample_images/sample_images_PA/test.png' --> is read as hwc 512,512,3
                            # image '/E/Sample_images/sample_images_FIJI/blobs.png' --> is read as hwc 1,254,256
                            # --> this discrepency of skimage makes no sense to me --> imageio does not male this error apparently but are there some specificieties ???
                    except Exception:
                        pass
            else:
                # load numpy image directly
                if f.lower().endswith('.npy'):
                    image = np.load(f)
                    try:
                        with open(f + '.meta') as json_file:
                            metadata = json.load(json_file)
                    except Exception:
                        logger.debug('could not load metadata ' + str(f + '.meta'))
                    # replace metadata from this file
                    return metadata, image
                else:
                    all_data = np.load(f)
                    image = all_data['data']
                    # Dirty way to recover first data in an image if data does not exist...
                    if image is None:
                        for dat in all_data:
                            image = dat
                            break
                    # TODO allow support for metadata some day
                    return None, image

        if voxel_x is not None and voxel_z is not None:
            ar = voxel_z / voxel_x

        logger.debug('original dimensions:' + str(image.shape))

        if image.shape[1] != height and image.ndim == 4 and t_frames is None:
            # print('I am called 1')
            image = np.moveaxis(image, [1], -1)

        if image.ndim >= 3 and image.shape[2] != height and image.ndim == 5:
            # print('I am called 2')
            image = np.moveaxis(image, [2], -1)

        if channels is not None and image.ndim == 3 and image.shape[0] == channels:
            # print('I am called 3')
            image = np.moveaxis(image, [0], -1)

        if channels is not None and image.ndim == 4 and image.shape[1] == channels:
            # print('I am called 4')
            image = np.moveaxis(image, [1], -1)

        dimensions_string += 'hw'

        # bug fix for images having d=1 incompatible with squeeze; need be done for every dimension by the way...
        if depth is not None and depth > 1:
            dimensions_string = 'd' + dimensions_string

        if channels is None and width != image.shape[-1] and len(image.shape) > 2:
            channels = image.shape[-1]

        if channels is not None and channels > 1:
            dimensions_string += 'c'

        # bug fix for images having t=1 incompatible with squeeze; need be done for every dimension by the way...
        if t_frames is not None and t_frames > 1:
            dimensions_string = 't' + dimensions_string
        else:
            if image.ndim > len(dimensions_string):
                dimensions_string = 't' + dimensions_string
                t_frames = image.shape[0]

        if width is None and image.ndim >= 3:
            width = image.shape[-2]
        if height is None and image.ndim >= 3:
            height = image.shape[-3]

        if width is None and image.ndim == 2:
            width = image.shape[-1]
        if height is None and image.ndim == 2:
            height = image.shape[-2]

        # update metadata
        metadata.update({'w': width, 'h': height, 'c': channels, 'd': depth, 't': t_frames, 'bits': bits, 'vx': voxel_x,
                         'vy': voxel_y, 'vz': voxel_z, 'AR': ar, 'dimensions': dimensions_string, 'LUTs': luts,
                         'times': times, 'Overlays': overlays, 'ROI': roi, 'timelapse': timelapse,
                         'creation_time': creation_time})
        # print(metadata)

        logger.debug('image params:' + str(metadata))
        logger.debug('final shape:' + str(image.shape))

        return metadata, image

    # def _fix_dimensions1(self):
    #     if

    def imageread(filePath):
        from batoolset.img import to_stack
        # TODO return other stuff here such as nb of frames ... do I need skimage to read or should I use smthg else
        # temp = skimage.io.imread(filePath[0])
        # h, w, c = temp.shape
        # d = len(filePath)
        # volume = np.zeros((d, w, h, c), dtype=temp.dtype)
        # rather do stack
        # k = 0
        # for img in filePath:  # assuming tif
        #     im = skimage.io.imread(img)
        #     volume[k, :, :, :] = np.swapaxes(im[:, :, :], 0, 1)
        #     k += 1
        # volume = np.stack([skimage.io.imread(img) for img in filePath],axis=0) # will take more memory but more elegant than before
        return to_stack(filePath)


