'''Regression tests for batoolset.img

Focused on the Img class (numpy ndarray subclass) since it's the core object
almost everything else in the codebase is built on top of. Covers:
  - blank-image construction (incl. the metadata=None crash that was fixed)
  - has_dimension synonym handling (incl. the duplicate-definition bug that was fixed)
  - __array_finalize__ metadata/z_scale propagation through slicing, arithmetic and view-casting
  - rescale (module function and Img method) on the currently pinned skimage version
  - a save/load round trip
  - a handful of core module-level pure functions
  - __repr__/__str__ (re-added after __array_finalize__; see the RETRY NOTE in img.py
    right above Img.__repr__ for why this previously broke and is now guarded)
'''

import numpy as np
import pytest

from batoolset.img import Img, rescale, clip, invert, guess_dimensions


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_blank_image_construction_without_metadata():
    # regression: used to raise IndexError because args was empty and metadata
    # was None, so `isinstance(args[0], Img)` blew up
    im = Img(h=20, w=20, c=3, bits=8)
    assert im.shape == (20, 20, 3)
    assert im.metadata['dimensions'] == 'hwc'
    assert im.dtype == np.uint8


def test_blank_image_construction_with_explicit_metadata():
    im = Img(h=10, w=10, c=1, bits=8, metadata={'note': 'hello'})
    assert im.metadata['note'] == 'hello'
    assert im.metadata['dimensions'] == 'hwc'


@pytest.mark.parametrize('bits,expected_dtype', [(8, np.uint8), (16, np.uint16), (32, np.float32)])
def test_blank_image_dtype_matches_requested_bits(bits, expected_dtype):
    im = Img(h=5, w=5, bits=bits)
    assert im.dtype == expected_dtype


def test_image_from_ndarray_gets_metadata():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    im = Img(arr)
    assert isinstance(im, Img)
    assert im.metadata is not None
    assert im.metadata['dimensions'] is None  # not guessed automatically here


def test_image_from_existing_img_recovers_its_metadata():
    im1 = Img(h=8, w=8, c=3, bits=8)
    im1.metadata['tag'] = 'from_im1'
    im2 = Img(im1)
    assert im2.metadata['tag'] == 'from_im1'


def test_image_from_path_list_and_glob(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f'frame_{i}.tif'
        frame = Img(h=6, w=6, bits=8)
        frame[:] = i
        frame.save(str(p))
        paths.append(str(p))

    stack_from_list = Img(paths)
    assert stack_from_list.shape == (3, 6, 6)

    stack_from_glob = Img(str(tmp_path / '*.tif'))
    assert stack_from_glob.shape == (3, 6, 6)


def test_load_missing_file_raises_clear_error_not_unboundlocalerror():
    # regression: the PIL fallback path used to be unreachable, so a failed
    # load would previously blow up with UnboundLocalError on `meta` instead
    # of a meaningful error
    with pytest.raises(Exception) as excinfo:
        Img('this_file_does_not_exist_at_all_xyz.tif')
    assert not isinstance(excinfo.value, UnboundLocalError)


def test_new_raises_instead_of_returning_none_for_unsupported_args():
    # regression: __new__ used to log a critical error and silently return None
    # for arguments it couldn't handle, forcing every caller to remember an
    # `is None` check; it should now raise instead
    with pytest.raises(Exception):
        Img(12345)  # not a str/list/ndarray/Img and no dimension kwargs -> unsupported


def test_default_z_scale_is_set_on_fresh_image():
    # regression: z_scale was never initialized in __new__, so get_d_scaling()
    # raised AttributeError on a freshly constructed image
    im = Img(h=5, w=5, c=3)
    assert im.get_d_scaling() == 1.0


# ---------------------------------------------------------------------------
# has_dimension (synonyms + duplicate-definition regression)
# ---------------------------------------------------------------------------

def test_has_dimension_direct_names():
    im = Img(h=4, w=4, c=3)
    assert im.has_dimension('h') is True
    assert im.has_dimension('w') is True
    assert im.has_dimension('c') is True
    assert im.has_dimension('d') is False
    assert im.has_dimension('t') is False


def test_has_dimension_synonyms():
    im = Img(h=4, w=4, c=3)
    # x/y/z are synonyms for w/h/d
    assert im.has_dimension('x') is True
    assert im.has_dimension('y') is True
    assert im.has_dimension('z') is False  # no d dimension here


def test_has_dimension_only_defined_once():
    # regression: has_dimension used to be defined twice on the class, with the
    # dead first copy referencing a nonexistent self.meta_data attribute
    definitions = [name for name in vars(Img) if name == 'has_dimension']
    assert len(definitions) == 1
    assert Img.has_dimension.__qualname__ == 'Img.has_dimension'


# ---------------------------------------------------------------------------
# __array_finalize__ (metadata / z_scale propagation)
# ---------------------------------------------------------------------------

def test_slicing_preserves_metadata():
    im = Img(h=20, w=20, c=3, bits=8)
    im.metadata['tag'] = 'orig'
    sub = im[0:10, 0:10]
    assert isinstance(sub, Img)
    assert sub.metadata is not None
    assert sub.metadata['tag'] == 'orig'


def test_arithmetic_preserves_metadata():
    im = Img(h=5, w=5, c=3, bits=8)
    im.metadata['tag'] = 'orig'
    added = im + 1
    assert isinstance(added, Img)
    assert added.metadata['tag'] == 'orig'


def test_slice_metadata_is_independent_copy():
    im = Img(h=20, w=20, c=3, bits=8)
    im.metadata['tag'] = 'orig'
    sub = im[0:10, 0:10]
    sub.metadata['tag'] = 'child'
    assert im.metadata['tag'] == 'orig'
    assert sub.metadata['tag'] == 'child'


def test_slice_LUTs_list_is_not_shared_with_parent():
    # regression: __array_finalize__ used to only shallow-copy the metadata dict,
    # so a mutable value like 'LUTs' (a list) was the *same* list object on the
    # parent and every derived view/slice -- mutating one silently mutated all
    im = Img(h=20, w=20, c=3, bits=8)
    im.metadata['LUTs'] = ['gray', 'red', 'green']
    sub = im[0:10, 0:10]
    sub.metadata['LUTs'].append('blue')
    assert im.metadata['LUTs'] == ['gray', 'red', 'green']
    assert sub.metadata['LUTs'] == ['gray', 'red', 'green', 'blue']


def test_view_casting_plain_ndarray_does_not_crash():
    plain = np.zeros((5, 5), dtype=np.uint8).view(Img)
    # no source metadata to propagate -> None, but accessing it must not raise
    assert plain.metadata is None


def test_z_scale_propagates_through_slicing():
    im = Img(h=10, w=10, c=3, bits=8)
    im.set_d_scaling(2.5)
    sub = im[0:5, 0:5]
    assert sub.get_d_scaling() == 2.5


# ---------------------------------------------------------------------------
# rescale (skimage multichannel -> channel_axis regression)
# ---------------------------------------------------------------------------

def test_module_rescale_runs_and_halves_size():
    im = Img(h=20, w=20, c=3, bits=8).astype(np.float32)
    out = rescale(im, scaling_factor=2)
    assert out.shape[:2] == (10, 10)


def test_img_method_rescale_runs_and_halves_size():
    im = Img(h=20, w=20, c=3, bits=8).astype(np.float32)
    out = im.rescale(factor=2)
    assert out.shape[:2] == (10, 10)


# ---------------------------------------------------------------------------
# __repr__ / __str__
# ---------------------------------------------------------------------------

def test_repr_and_str_basic():
    im = Img(h=10, w=10, c=3, bits=8)
    im.metadata['path'] = '/some/file.tif'
    text = repr(im)
    assert 'shape=(10, 10, 3)' in text
    assert 'dimensions' in text
    assert '/some/file.tif' in text
    assert str(im) == repr(im)


def test_repr_on_large_image_triggers_numpy_summarization_path_safely():
    # regression: this is the scenario that likely broke an earlier __str__ attempt --
    # numpy internally builds summarized edge-slices (new Img instances) of large arrays
    # when formatting them; without __array_finalize__ those had no .metadata and crashed
    big = Img(h=2000, w=2000, c=3, bits=8)
    assert 'shape=(2000, 2000, 3)' in repr(big)
    assert 'shape=(2000, 2000, 3)' in str(big)


def test_repr_on_nested_slices_and_arithmetic_results():
    big = Img(h=200, w=200, c=3, bits=8)
    sub = big[50:150, 50:150]
    deep = sub[0:5, 0:5]
    added = big + 1
    for arr in (sub, deep, added):
        assert repr(arr)  # must not raise


def test_repr_on_metadata_less_view_falls_back_gracefully():
    plain = np.zeros((5, 5), dtype=np.uint8).view(Img)
    text = repr(plain)  # must not raise even though .metadata is None
    assert 'shape=(5, 5)' in text


def test_to_string_unaffected_by_repr_str():
    im = Img(h=10, w=10, c=3, bits=8)
    assert 'Image:' in im.to_string()


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------

def test_save_and_reload_tiff_round_trip(tmp_path):
    im = Img(h=16, w=16, c=3, bits=8)
    im[:] = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(im.shape)
    out_path = str(tmp_path / 'roundtrip.tif')
    im.save(out_path)
    reloaded = Img(out_path)
    assert reloaded.shape == im.shape
    assert np.array_equal(np.asarray(reloaded), np.asarray(im))


# ---------------------------------------------------------------------------
# A handful of core pure functions
# ---------------------------------------------------------------------------

def test_guess_dimensions_2d():
    assert guess_dimensions(np.zeros((10, 10))) == 'hw'


def test_guess_dimensions_3d_hwc_vs_dhw():
    assert guess_dimensions(np.zeros((10, 10, 3))) == 'hwc'  # last axis smallest -> channels
    assert guess_dimensions(np.zeros((5, 10, 10))) == 'dhw'  # last axis not smallest -> z-stack


def test_clip_with_tuple():
    img = np.array([[0.5, 1.2, 0.8], [1.9, 2.5, 1.0], [0.3, 0.7, 1.5]])
    out = clip(img, tuple=(0.5, 1.5))
    assert out.min() >= 0.5
    assert out.max() <= 1.5


def test_invert_boolean():
    img = np.array([[True, False], [False, True]])
    out = invert(img)
    assert np.array_equal(out, np.array([[False, True], [True, False]]))


def test_invert_numeric():
    img = np.array([0, 1, 2, 3], dtype=np.uint8)
    out = invert(img)
    # min becomes max and vice versa
    assert out.max() == 3
    assert out.min() == 0
    assert np.array_equal(out, np.array([3, 2, 1, 0]))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
