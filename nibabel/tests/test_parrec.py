""" Testing parrec module
"""

from os.path import join as pjoin, dirname, basename
from glob import glob
from warnings import simplefilter

import numpy as np
from numpy import array as npa

from .. import parrec
from ..parrec import (parse_PAR_header, PARRECHeader, PARRECError, vol_numbers,
                      vol_is_full, PARRECImage, PARRECArrayProxy)
from ..openers import Opener
from ..fileholders import FileHolder

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..testing import catch_warn_reset, suppress_warnings


DATA_PATH = pjoin(dirname(__file__), 'data')
EG_PAR = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.PAR')
EG_REC = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.REC')
with Opener(EG_PAR, 'rt') as _fobj:
    HDR_INFO, HDR_DEFS = parse_PAR_header(_fobj)
# Fake truncated
TRUNC_PAR = pjoin(DATA_PATH, 'phantom_truncated.PAR')
TRUNC_REC = pjoin(DATA_PATH, 'phantom_truncated.REC')
# Affine as we determined it mid-2014
AN_OLD_AFFINE = np.array(
    [[-3.64994708, 0.,   1.83564171, 123.66276611],
     [0.,         -3.75, 0.,          115.617    ],
     [0.86045705,  0.,   7.78655376, -27.91161211],
     [0.,          0.,   0.,           1.        ]])
# Affine from Philips-created NIfTI
PHILIPS_AFFINE = np.array(
    [[  -3.65  ,   -0.0016,    1.8356,  125.4881],
     [   0.0016,   -3.75  ,   -0.0004,  117.4916],
     [   0.8604,    0.0002,    7.7866,  -28.3411],
     [   0.    ,    0.    ,    0.    ,    1.    ]])

# Affines generated by parrec.py from test data in many orientations
# Data from http://psydata.ovgu.de/philips_achieva_testfiles/conversion2
PREVIOUS_AFFINES={
    "Phantom_EPI_3mm_cor_20APtrans_15RLrot_SENSE_15_1" :
    npa([[  -3.        ,    0.        ,    0.        ,  118.5       ],
         [   0.        ,   -0.77645714,   -3.18755523,   72.82738377],
         [   0.        ,   -2.89777748,    0.85410285,   97.80720486],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_cor_SENSE_8_1" :
    npa([[  -3.  ,    0.  ,    0.  ,  118.5 ],
         [   0.  ,    0.  ,   -3.3 ,   64.35],
         [   0.  ,   -3.  ,    0.  ,  118.5 ],
         [   0.  ,    0.  ,    0.  ,    1.  ]]),
    "Phantom_EPI_3mm_sag_15AP_SENSE_13_1" :
    npa([[   0.        ,    0.77645714,    3.18755523,  -92.82738377],
         [  -3.        ,    0.        ,    0.        ,  118.5       ],
         [   0.        ,   -2.89777748,    0.85410285,   97.80720486],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_sag_15FH_SENSE_12_1" :
    npa([[   0.77645714,    0.        ,    3.18755523,  -92.82738377],
         [  -2.89777748,    0.        ,    0.85410285,   97.80720486],
         [   0.        ,   -3.        ,    0.        ,  118.5       ],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_sag_15RL_SENSE_11_1" :
    npa([[   0.        ,    0.        ,    3.3       ,  -64.35      ],
         [  -2.89777748,   -0.77645714,    0.        ,  145.13226726],
         [   0.77645714,   -2.89777748,    0.        ,   83.79215357],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_sag_SENSE_7_1" :
    npa([[   0.  ,    0.  ,    3.3 ,  -64.35],
         [  -3.  ,    0.  ,    0.  ,  118.5 ],
         [   0.  ,   -3.  ,    0.  ,  118.5 ],
         [   0.  ,    0.  ,    0.  ,    1.  ]]),
    "Phantom_EPI_3mm_tra_-30AP_10RL_20FH_SENSE_14_1" :
    npa([[   0.  ,    0.  ,    3.3 ,  -74.35],
         [  -3.  ,    0.  ,    0.  ,  148.5 ],
         [   0.  ,   -3.  ,    0.  ,  138.5 ],
         [   0.  ,    0.  ,    0.  ,    1.  ]]),
    "Phantom_EPI_3mm_tra_15FH_SENSE_9_1" :
    npa([[   0.77645714,    0.        ,    3.18755523,  -92.82738377],
         [  -2.89777748,    0.        ,    0.85410285,   97.80720486],
         [   0.        ,   -3.        ,    0.        ,  118.5       ],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_tra_15RL_SENSE_10_1" :
    npa([[   0.        ,    0.        ,    3.3       ,  -64.35      ],
         [  -2.89777748,   -0.77645714,    0.        ,  145.13226726],
         [   0.77645714,   -2.89777748,    0.        ,   83.79215357],
         [   0.        ,    0.        ,    0.        ,    1.        ]]),
    "Phantom_EPI_3mm_tra_SENSE_6_1" :
    npa([[  -3.  ,    0.  ,    0.  ,  118.5 ],
         [   0.  ,   -3.  ,    0.  ,  118.5 ],
         [   0.  ,    0.  ,    3.3 ,  -64.35],
         [   0.  ,    0.  ,    0.  ,    1.  ]]),
}

# Original values for b values in DTI.PAR, still in PSL orientation
DTI_PAR_BVECS = np.array([[-0.667,  -0.667,  -0.333],
                          [-0.333,   0.667,  -0.667],
                          [-0.667,   0.333,   0.667],
                          [-0.707,  -0.000,  -0.707],
                          [-0.707,   0.707,   0.000],
                          [-0.000,   0.707,   0.707],
                          [ 0.000,   0.000,   0.000],
                          [ 0.000,   0.000,   0.000]])

# DTI.PAR values for bvecs
DTI_PAR_BVALS = [1000] * 6 + [0, 1000]


def test_header():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_equal(hdr.get_data_shape(), (64, 64, 9, 3))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.int16))
    assert_equal(hdr.get_zooms(), (3.75, 3.75, 8.0, 2.0))
    assert_equal(hdr.get_data_offset(), 0)
    si = np.array([np.unique(x) for x in hdr.get_data_scaling()]).ravel()
    assert_almost_equal(si, (1.2903541326522827, 0.0), 5)


def test_header_scaling():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    def_scaling = [np.unique(x) for x in hdr.get_data_scaling()]
    fp_scaling = [np.unique(x) for x in hdr.get_data_scaling('fp')]
    dv_scaling = [np.unique(x) for x in hdr.get_data_scaling('dv')]
    # Check default is dv scaling
    assert_array_equal(def_scaling, dv_scaling)
    # And that it's almost the same as that from the converted nifti
    assert_almost_equal(dv_scaling, [[1.2903541326522827], [0.0]], 5)
    # Check that default for get_slope_inter is dv scaling
    for hdr in (hdr, PARRECHeader(HDR_INFO, HDR_DEFS)):
        scaling = [np.unique(x) for x in hdr.get_data_scaling()]
        assert_array_equal(scaling, dv_scaling)
    # Check we can change the default
    assert_false(np.all(fp_scaling == dv_scaling))


def test_orientation():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_array_equal(HDR_DEFS['slice orientation'], 1)
    assert_equal(hdr.get_slice_orientation(), 'transverse')
    hdr_defc = hdr.image_defs
    hdr_defc['slice orientation'] = 2
    assert_equal(hdr.get_slice_orientation(), 'sagittal')
    hdr_defc['slice orientation'] = 3
    assert_equal(hdr.get_slice_orientation(), 'coronal')


def test_data_offset():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_equal(hdr.get_data_offset(), 0)
    # Can set 0
    hdr.set_data_offset(0)
    # Can't set anything else
    assert_raises(PARRECError, hdr.set_data_offset, 1)


def test_affine():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    default = hdr.get_affine()
    scanner = hdr.get_affine(origin='scanner')
    fov = hdr.get_affine(origin='fov')
    assert_array_equal(default, scanner)
    # rotation part is same
    assert_array_equal(scanner[:3, :3], fov[:3, :3])
    # offset not
    assert_false(np.all(scanner[:3, 3] == fov[:3, 3]))
    # Regression test against what we were getting before
    assert_almost_equal(default, AN_OLD_AFFINE)
    # Test against RZS of Philips affine
    assert_almost_equal(default[:3, :3], PHILIPS_AFFINE[:3, :3], 2)


def test_affine_regression():
    # Test against checked affines from previous runs
    # Checked against Michael's data using some GUI tools
    # Data at http://psydata.ovgu.de/philips_achieva_testfiles/conversion2
    for basename, exp_affine in PREVIOUS_AFFINES.items():
        fname = pjoin(DATA_PATH, basename + '.PAR')
        with open(fname, 'rt') as fobj:
            hdr = PARRECHeader.from_fileobj(fobj)
        assert_almost_equal(hdr.get_affine(), exp_affine)


def test_get_voxel_size_deprecated():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    with catch_warn_reset(modules=[parrec], record=True) as wlist:
        simplefilter('always')
        hdr.get_voxel_size()
    assert_equal(wlist[0].category, DeprecationWarning)


def test_get_sorted_slice_indices():
    # Test sorted slice indices
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    n_slices = len(HDR_DEFS)
    assert_array_equal(hdr.get_sorted_slice_indices(), range(n_slices))
    # Reverse - volume order preserved
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS[::-1])
    assert_array_equal(hdr.get_sorted_slice_indices(),
                       [8,  7,  6,  5,  4,  3,  2,  1,  0,
                        17, 16, 15, 14, 13, 12, 11, 10, 9,
                        26, 25, 24, 23, 22, 21, 20, 19, 18])
    # Omit last slice, only two volumes
    with catch_warn_reset(modules=[parrec], record=True):
        hdr = PARRECHeader(HDR_INFO, HDR_DEFS[:-1], permit_truncated=True)
    assert_array_equal(hdr.get_sorted_slice_indices(), range(n_slices - 9))


def test_vol_number():
    # Test algorithm for calculating volume number
    assert_array_equal(vol_numbers([1, 3, 0]), [0, 0, 0])
    assert_array_equal(vol_numbers([1, 3, 0, 0]), [ 0, 0, 0, 1])
    assert_array_equal(vol_numbers([1, 3, 0, 0, 0]), [0, 0, 0, 1, 2])
    assert_array_equal(vol_numbers([1, 3, 0, 0, 4]), [0, 0, 0, 1, 0])
    assert_array_equal(vol_numbers([1, 3, 0, 3, 1, 0]),
                       [0, 0, 0, 1, 1, 1])
    assert_array_equal(vol_numbers([1, 3, 0, 3, 1, 0, 4]),
                       [0, 0, 0, 1, 1, 1, 0])
    assert_array_equal(vol_numbers([1, 3, 0, 3, 1, 0, 3, 1, 0]),
                       [0, 0, 0, 1, 1, 1, 2, 2, 2])


def test_vol_is_full():
    assert_array_equal(vol_is_full([3, 2, 1], 3), True)
    assert_array_equal(vol_is_full([3, 2, 1], 4), False)
    assert_array_equal(vol_is_full([4, 2, 1], 4), False)
    assert_array_equal(vol_is_full([3, 2, 4, 1], 4), True)
    assert_array_equal(vol_is_full([3, 2, 1], 3, 0), False)
    assert_array_equal(vol_is_full([3, 2, 0, 1], 3, 0), True)
    assert_raises(ValueError, vol_is_full, [2, 1, 0], 2)
    assert_raises(ValueError, vol_is_full, [3, 2, 1], 3, 2)
    assert_array_equal(vol_is_full([3, 2, 1, 2, 3, 1], 3),
                       [True] * 6)
    assert_array_equal(vol_is_full([3, 2, 1, 2, 3], 3),
                       [True, True, True, False, False])


def gen_par_fobj():
    for par in glob(pjoin(DATA_PATH, '*.PAR')):
        with open(par, 'rt') as fobj:
            yield par, fobj


def test_truncated_load():
    # Test loading of truncated header
    with open(TRUNC_PAR, 'rt') as fobj:
        gen_info, slice_info = parse_PAR_header(fobj)
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    with catch_warn_reset(record=True) as wlist:
        hdr = PARRECHeader(gen_info, slice_info, True)
        assert_equal(len(wlist), 1)


def test_vol_calculations():
    # Test vol_is_full on sample data
    for par, fobj in gen_par_fobj():
        gen_info, slice_info = parse_PAR_header(fobj)
        slice_nos = slice_info['slice number']
        max_slice = gen_info['max_slices']
        assert_equal(set(slice_nos), set(range(1, max_slice + 1)))
        assert_array_equal(vol_is_full(slice_nos, max_slice), True)
        if par.endswith('NA.PAR'):
            continue # Cannot parse this one
        # Load truncated without warnings
        with suppress_warnings():
            hdr = PARRECHeader(gen_info, slice_info, True)
        # Fourth dimension shows same number of volumes as vol_numbers
        shape = hdr.get_data_shape()
        d4 = 1 if len(shape) == 3 else shape[3]
        assert_equal(max(vol_numbers(slice_nos)), d4 - 1)


def test_diffusion_parameters():
    # Check getting diffusion parameters from diffusion example
    dti_par = pjoin(DATA_PATH, 'DTI.PAR')
    with open(dti_par, 'rt') as fobj:
        dti_hdr = PARRECHeader.from_fileobj(fobj)
    assert_equal(dti_hdr.get_data_shape(), (80, 80, 10, 8))
    assert_equal(dti_hdr.general_info['diffusion'], 1)
    bvals, bvecs = dti_hdr.get_bvals_bvecs()
    assert_almost_equal(bvals, DTI_PAR_BVALS)
    # DTI_PAR_BVECS gives bvecs copied from first slice each vol in DTI.PAR
    # Permute to match bvec directions to acquisition directions
    assert_almost_equal(bvecs, DTI_PAR_BVECS[:, [2, 0, 1]])
    # Check q vectors
    assert_almost_equal(dti_hdr.get_q_vectors(), bvals[:, None] * bvecs)


def test_null_diffusion_params():
    # Test non-diffusion PARs return None for diffusion params
    for par, fobj in gen_par_fobj():
        if basename(par) in ('DTI.PAR', 'NA.PAR'):
            continue
        gen_info, slice_info = parse_PAR_header(fobj)
        with suppress_warnings():
            hdr = PARRECHeader(gen_info, slice_info, True)
        assert_equal(hdr.get_bvals_bvecs(), (None, None))
        assert_equal(hdr.get_q_vectors(), None)


def test_epi_params():
    # Check EPI conversion
    for par_root in ('T2_-interleaved', 'T2_', 'phantom_EPI_asc_CLEAR_2_1'):
        epi_par = pjoin(DATA_PATH, par_root + '.PAR')
        with open(epi_par, 'rt') as fobj:
            epi_hdr = PARRECHeader.from_fileobj(fobj)
        assert_equal(len(epi_hdr.get_data_shape()), 4)
        assert_almost_equal(epi_hdr.get_zooms()[-1], 2.0)


def test_truncations():
    # Test tests for truncation
    par = pjoin(DATA_PATH, 'T2_.PAR')
    with open(par, 'rt') as fobj:
        gen_info, slice_info = parse_PAR_header(fobj)
    # Header is well-formed as is
    hdr = PARRECHeader(gen_info, slice_info)
    assert_equal(hdr.get_data_shape(), (80, 80, 10, 2))
    # Drop one line, raises error
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info[:-1])
    # When we are permissive, we raise a warning, and drop a volume
    with catch_warn_reset(modules=[parrec], record=True) as wlist:
        hdr = PARRECHeader(gen_info, slice_info[:-1], permit_truncated=True)
        assert_equal(len(wlist), 1)
    assert_equal(hdr.get_data_shape(), (80, 80, 10))
    # Increase max slices to raise error
    gen_info['max_slices'] = 11
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    gen_info['max_slices'] = 10
    hdr = PARRECHeader(gen_info, slice_info)
    # Increase max_echoes
    gen_info['max_echoes'] = 2
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    gen_info['max_echoes'] = 1
    hdr = PARRECHeader(gen_info, slice_info)
    # dyamics
    gen_info['max_dynamics'] = 3
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    gen_info['max_dynamics'] = 2
    hdr = PARRECHeader(gen_info, slice_info)
    # number of b values
    gen_info['max_diffusion_values'] = 2
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    gen_info['max_diffusion_values'] = 1
    hdr = PARRECHeader(gen_info, slice_info)
    # number of unique gradients
    gen_info['max_gradient_orient'] = 2
    assert_raises(PARRECError, PARRECHeader, gen_info, slice_info)
    gen_info['max_gradient_orient'] = 1
    hdr = PARRECHeader(gen_info, slice_info)


def test__get_uniqe_image_defs():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS.copy())
    uip = hdr._get_unique_image_prop
    assert_equal(uip('image pixel size'), 16)
    # Make values not same - raise error
    hdr.image_defs['image pixel size'][3] = 32
    assert_raises(PARRECError, uip, 'image pixel size')
    assert_array_equal(uip('recon resolution'), [64, 64])
    hdr.image_defs['recon resolution'][4, 1] = 32
    assert_raises(PARRECError, uip, 'recon resolution')
    assert_array_equal(uip('image angulation'), [-13.26, 0, 0])
    hdr.image_defs['image angulation'][5, 2] = 1
    assert_raises(PARRECError, uip, 'image angulation')
    # This one differs from the outset
    assert_raises(PARRECError, uip, 'slice number')


def test_copy_on_init():
    # Test that input dict / array gets copied when making header
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert_false(hdr.general_info is HDR_INFO)
    hdr.general_info['max_slices'] = 10
    assert_equal(hdr.general_info['max_slices'], 10)
    assert_equal(HDR_INFO['max_slices'], 9)
    assert_false(hdr.image_defs is HDR_DEFS)
    hdr.image_defs['image pixel size'] = 8
    assert_array_equal(hdr.image_defs['image pixel size'], 8)
    assert_array_equal(HDR_DEFS['image pixel size'], 16)


def assert_arr_dict_equal(dict1, dict2):
    assert_equal(set(dict1), set(dict2))
    for key, value1 in dict1.items():
        value2 = dict2[key]
        assert_array_equal(value1, value2)


def test_header_copy():
    # Test header copying
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    hdr2 = hdr.copy()

    def assert_copy_ok(hdr1, hdr2):
        assert_false(hdr1 is hdr2)
        assert_equal(hdr1.permit_truncated, hdr2.permit_truncated)
        assert_false(hdr1.general_info is hdr2.general_info)
        assert_arr_dict_equal(hdr1.general_info, hdr2.general_info)
        assert_false(hdr1.image_defs is hdr2.image_defs)
        assert_array_equal(hdr1.image_defs, hdr2.image_defs)

    assert_copy_ok(hdr, hdr2)
    assert_false(hdr.permit_truncated)
    assert_false(hdr2.permit_truncated)
    with open(TRUNC_PAR, 'rt') as fobj:
        assert_raises(PARRECError, PARRECHeader.from_fileobj, fobj)
    with open(TRUNC_PAR, 'rt') as fobj:
        trunc_hdr = PARRECHeader.from_fileobj(fobj, True)
    assert_true(trunc_hdr.permit_truncated)
    trunc_hdr2 = trunc_hdr.copy()
    assert_copy_ok(trunc_hdr, trunc_hdr2)


def test_image_creation():
    # Test parts of image API in parrec image creation
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    arr_prox_dv = np.array(PARRECArrayProxy(EG_REC, hdr, 'dv'))
    arr_prox_fp = np.array(PARRECArrayProxy(EG_REC, hdr, 'fp'))
    good_map = dict(image = FileHolder(EG_REC),
                    header = FileHolder(EG_PAR))
    trunc_map = dict(image = FileHolder(TRUNC_REC),
                     header = FileHolder(TRUNC_PAR))
    for func, good_param, trunc_param in (
        (PARRECImage.from_filename, EG_PAR, TRUNC_PAR),
        (PARRECImage.load, EG_PAR, TRUNC_PAR),
        (parrec.load, EG_PAR, TRUNC_PAR),
        (PARRECImage.from_file_map, good_map, trunc_map)):
        img = func(good_param)
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(good_param, False)
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(good_param, False, 'dv')
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(good_param, False, 'fp')
        assert_array_equal(img.dataobj, arr_prox_fp)
        assert_raises(PARRECError, func, trunc_param)
        assert_raises(PARRECError, func, trunc_param, False)
        img = func(trunc_param, True)
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(trunc_param, True, 'dv')
        assert_array_equal(img.dataobj, arr_prox_dv)
        img = func(trunc_param, True, 'fp')
        assert_array_equal(img.dataobj, arr_prox_fp)
