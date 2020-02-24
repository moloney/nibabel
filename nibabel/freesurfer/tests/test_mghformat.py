# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for mghformat reading writing'''

import os
import io

import numpy as np

from .. import load, save
from ...openers import ImageOpener
from ..mghformat import MGHHeader, MGHError, MGHImage
from ...tmpdirs import InTemporaryDirectory
from ...fileholders import FileHolder
from ...spatialimages import HeaderDataError
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from ... import imageglobals


import pytest

from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from ...testing import data_path

from ...tests import test_spatialimages as tsi
from ...tests.test_wrapstruct import _TestLabeledWrapStruct

MGZ_FNAME = os.path.join(data_path, 'test.mgz')

# sample voxel to ras matrix (mri_info --vox2ras)
v2r = np.array([[1, 2, 3, -13], [2, 3, 1, -11.5],
                [3, 1, 2, -11.5], [0, 0, 0, 1]], dtype=np.float32)
# sample voxel to ras - tkr matrix (mri_info --vox2ras-tkr)
v2rtkr = np.array([[-1.0, 0.0, 0.0, 1.5],
                   [0.0, 0.0, 1.0, -2.5],
                   [0.0, -1.0, 0.0, 2.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

BIG_CODES = ('>', 'big', 'BIG', 'b', 'be', 'B', 'BE')
LITTLE_CODES = ('<', 'little', 'l', 'le', 'L', 'LE')

if sys_is_le:
    BIG_CODES += ('swapped', 's', 'S', '!')
    LITTLE_CODES += ('native', 'n', 'N', '=', '|', 'i', 'I')
else:
    BIG_CODES += ('native', 'n', 'N', '=', '|', 'i', 'I')
    LITTLE_CODES += ('swapped', 's', 'S', '!')



def test_read_mgh():
    # test.mgz was generated by the following command
    # mri_volsynth --dim 3 4 5 2 --vol test.mgz
    # --cdircos 1 2 3 --rdircos 2 3 1 --sdircos 3 1 2
    # mri_volsynth is a FreeSurfer command
    mgz = load(MGZ_FNAME)

    # header
    h = mgz.header
    assert h['version'] == 1
    assert h['type'] == 3
    assert h['dof'] == 0
    assert h['goodRASFlag'] == 1
    assert_array_equal(h['dims'], [3, 4, 5, 2])
    assert_almost_equal(h['tr'], 2.0)
    assert_almost_equal(h['flip_angle'], 0.0)
    assert_almost_equal(h['te'], 0.0)
    assert_almost_equal(h['ti'], 0.0)
    assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 2])
    assert_array_almost_equal(h.get_vox2ras(), v2r)
    assert_array_almost_equal(h.get_vox2ras_tkr(), v2rtkr)

    # data. will be different for your own mri_volsynth invocation
    v = mgz.get_fdata()
    assert_almost_equal(v[1, 2, 3, 0], -0.3047, 4)
    assert_almost_equal(v[1, 2, 3, 1], 0.0018, 4)


def test_write_mgh():
    # write our data to a tmp file
    v = np.arange(120)
    v = v.reshape((5, 4, 3, 2)).astype(np.float32)
    # form a MGHImage object using data and vox2ras matrix
    img = MGHImage(v, v2r)
    with InTemporaryDirectory():
        save(img, 'tmpsave.mgz')
        # read from the tmp file and see if it checks out
        mgz = load('tmpsave.mgz')
        h = mgz.header
        dat = mgz.get_fdata()
        # Delete loaded image to allow file deletion by windows
        del mgz
    # header
    assert h['version'] == 1
    assert h['type'] == 3
    assert h['dof'] == 0
    assert h['goodRASFlag'] == 1
    assert np.array_equal(h['dims'], [5, 4, 3, 2])
    assert_almost_equal(h['tr'], 0.0)
    assert_almost_equal(h['flip_angle'], 0.0)
    assert_almost_equal(h['te'], 0.0)
    assert_almost_equal(h['ti'], 0.0)
    assert_almost_equal(h['fov'], 0.0)
    assert_array_almost_equal(h.get_vox2ras(), v2r)
    # data
    assert_almost_equal(dat, v, 7)


def test_write_noaffine_mgh():
    # now just save the image without the vox2ras transform
    # and see if it uses the default values to save
    v = np.ones((7, 13, 3, 22)).astype(np.uint8)
    # form a MGHImage object using data
    # and the default affine matrix (Note the "None")
    img = MGHImage(v, None)
    with InTemporaryDirectory():
        save(img, 'tmpsave.mgz')
        # read from the tmp file and see if it checks out
        mgz = load('tmpsave.mgz')
        h = mgz.header
        # Delete loaded image to allow file deletion by windows
        del mgz
    # header
    assert h['version'] == 1
    assert h['type'] == 0  # uint8 for mgh
    assert h['dof'] == 0
    assert h['goodRASFlag'] == 1
    assert np.array_equal(h['dims'], [7, 13, 3, 22])
    assert_almost_equal(h['tr'], 0.0)
    assert_almost_equal(h['flip_angle'], 0.0)
    assert_almost_equal(h['te'], 0.0)
    assert_almost_equal(h['ti'], 0.0)
    assert_almost_equal(h['fov'], 0.0)
    # important part -- whether default affine info is stored
    assert_array_almost_equal(h['Mdc'], [[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert_array_almost_equal(h['Pxyz_c'], [0, 0, 0])


def test_set_zooms():
    mgz = load(MGZ_FNAME)
    h = mgz.header
    assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 2])
    h.set_zooms([1, 1, 1, 3])
    assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 3])
    for zooms in ((-1, 1, 1, 1),
                  (1, -1, 1, 1),
                  (1, 1, -1, 1),
                  (1, 1, 1, -1),
                  (1, 1, 1, 1, 5)):
        with pytest.raises(HeaderDataError):
            h.set_zooms(zooms)
    # smoke test for tr=0
    h.set_zooms((1, 1, 1, 0))


def bad_dtype_mgh():
    ''' This function raises an MGHError exception because
    uint16 is not a valid MGH datatype.
    '''
    # try to write an unsigned short and make sure it
    # raises MGHError
    v = np.ones((7, 13, 3, 22)).astype(np.uint16)
    # form a MGHImage object using data
    # and the default affine matrix (Note the "None")
    MGHImage(v, None)


def test_bad_dtype_mgh():
    # Now test the above function
    with pytest.raises(MGHError):
        bad_dtype_mgh()


def test_filename_exts():
    # Test acceptable filename extensions
    v = np.ones((7, 13, 3, 22)).astype(np.uint8)
    # form a MGHImage object using data
    # and the default affine matrix (Note the "None")
    img = MGHImage(v, None)
    # Check if these extensions allow round trip
    for ext in ('.mgh', '.mgz'):
        with InTemporaryDirectory():
            fname = 'tmpname' + ext
            save(img, fname)
            # read from the tmp file and see if it checks out
            img_back = load(fname)
            assert_array_equal(img_back.get_fdata(), v)
            del img_back


def _mgh_rt(img, fobj):
    file_map = {'image': FileHolder(fileobj=fobj)}
    img.to_file_map(file_map)
    return MGHImage.from_file_map(file_map)


def test_header_updating():
    # Don't update the header information if the affine doesn't change.
    # Luckily the test.mgz dataset had a bad set of cosine vectors, so these
    # will be changed if the affine gets updated
    mgz = load(MGZ_FNAME)
    hdr = mgz.header
    # Test against mri_info output
    exp_aff = np.loadtxt(io.BytesIO(b"""
    1.0000   2.0000   3.0000   -13.0000
    2.0000   3.0000   1.0000   -11.5000
    3.0000   1.0000   2.0000   -11.5000
    0.0000   0.0000   0.0000     1.0000"""))
    assert_almost_equal(mgz.affine, exp_aff, 6)
    assert_almost_equal(hdr.get_affine(), exp_aff, 6)
    # Test that initial wonky header elements have not changed
    assert np.all(hdr['delta'] == 1)
    assert_almost_equal(hdr['Mdc'].T, exp_aff[:3, :3])
    # Save, reload, same thing
    img_fobj = io.BytesIO()
    mgz2 = _mgh_rt(mgz, img_fobj)
    hdr2 = mgz2.header
    assert_almost_equal(hdr2.get_affine(), exp_aff, 6)
    assert_array_equal(hdr2['delta'],1)
    # Change affine, change underlying header info
    exp_aff_d = exp_aff.copy()
    exp_aff_d[0, -1] = -14
    # This will (probably) become part of the official API
    mgz2._affine[:] = exp_aff_d
    mgz2.update_header()
    assert_almost_equal(hdr2.get_affine(), exp_aff_d, 6)
    RZS = exp_aff_d[:3, :3]
    assert_almost_equal(hdr2['delta'], np.sqrt(np.sum(RZS ** 2, axis=0)))
    assert_almost_equal(hdr2['Mdc'].T, RZS / hdr2['delta'])


def test_cosine_order():
    # Test we are interpreting the cosine order right
    data = np.arange(60).reshape((3, 4, 5)).astype(np.int32)
    aff = np.diag([2., 3, 4, 1])
    aff[0] = [2, 1, 0, 10]
    img = MGHImage(data, aff)
    assert_almost_equal(img.affine, aff, 6)
    img_fobj = io.BytesIO()
    img2 = _mgh_rt(img, img_fobj)
    hdr2 = img2.header
    RZS = aff[:3, :3]
    zooms = np.sqrt(np.sum(RZS ** 2, axis=0))
    assert_almost_equal(hdr2['Mdc'].T, RZS / zooms)
    assert_almost_equal(hdr2['delta'], zooms)


def test_eq():
    # Test headers compare properly
    hdr = MGHHeader()
    hdr2 = MGHHeader()
    assert hdr == hdr2
    hdr.set_data_shape((2, 3, 4))
    assert(hdr != hdr2)
    hdr2.set_data_shape((2, 3, 4))
    assert hdr == hdr2


def test_header_slope_inter():
    # Test placeholder slope / inter method
    hdr = MGHHeader()
    assert hdr.get_slope_inter() == (None, None)


def test_mgh_load_fileobj():
    # Checks the filename gets passed to array proxy
    #
    # This is a bit of an implementation detail, but the test is to make sure
    # that we aren't passing ImageOpener objects to the array proxy, as these
    # were confusing mmap on Python 3.  If there's some sensible reason not to
    # pass the filename to the array proxy, please feel free to change this
    # test.
    img = MGHImage.load(MGZ_FNAME)
    assert img.dataobj.file_like == MGZ_FNAME
    # Check fileobj also passed into dataobj
    with ImageOpener(MGZ_FNAME) as fobj:
        contents = fobj.read()
    bio = io.BytesIO(contents)
    fm = MGHImage.make_file_map(mapping=dict(image=bio))
    img2 = MGHImage.from_file_map(fm)
    assert(img2.dataobj.file_like is bio)
    assert_array_equal(img.get_fdata(), img2.get_fdata())


def test_mgh_affine_default():
    hdr = MGHHeader()
    hdr['goodRASFlag'] = 0
    hdr2 = MGHHeader(hdr.binaryblock)
    assert hdr2['goodRASFlag'] == 1
    assert_array_equal(hdr['Mdc'], hdr2['Mdc'])
    assert_array_equal(hdr['Pxyz_c'], hdr2['Pxyz_c'])


def test_mgh_set_data_shape():
    hdr = MGHHeader()
    hdr.set_data_shape((5,))
    assert_array_equal(hdr.get_data_shape(), (5, 1, 1))
    hdr.set_data_shape((5, 4))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 1))
    hdr.set_data_shape((5, 4, 3))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 3))
    hdr.set_data_shape((5, 4, 3, 2))
    assert_array_equal(hdr.get_data_shape(), (5, 4, 3, 2))
    with pytest.raises(ValueError):
        hdr.set_data_shape((5, 4, 3, 2, 1))


def test_mghheader_default_structarr():
    hdr = MGHHeader.default_structarr()
    assert hdr['version'] == 1
    assert_array_equal(hdr['dims'], 1)
    assert hdr['type'] == 3
    assert hdr['dof'] == 0
    assert hdr['goodRASFlag'] == 1
    assert_array_equal(hdr['delta'], 1)
    assert_array_equal(hdr['Mdc'], [[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert_array_equal(hdr['Pxyz_c'], 0)
    assert hdr['tr'] == 0
    assert hdr['flip_angle'] == 0
    assert hdr['te'] == 0
    assert hdr['ti'] == 0
    assert hdr['fov'] == 0

    for endianness in (None,) + BIG_CODES:
        hdr2 = MGHHeader.default_structarr(endianness=endianness)
        assert hdr2 == hdr
        assert hdr2.newbyteorder('>') == hdr

    for endianness in LITTLE_CODES:
        with pytest.raises(ValueError):
            MGHHeader.default_structarr(endianness=endianness)


def test_deprecated_fields():
    hdr = MGHHeader()
    hdr_data = MGHHeader._HeaderData(hdr.structarr)

    # mrparams is the only deprecated field at the moment
    # Accessing hdr_data is equivalent to accessing hdr, so double all checks
    assert_array_equal(hdr['mrparams'], 0)
    assert_array_equal(hdr_data['mrparams'], 0)

    hdr['mrparams'] = [1, 2, 3, 4]
    assert_array_almost_equal(hdr['mrparams'], [1, 2, 3, 4])
    assert hdr['tr'] == 1
    assert hdr['flip_angle'] == 2
    assert hdr['te'] == 3
    assert hdr['ti'] == 4
    assert hdr['fov'] == 0
    assert_array_almost_equal(hdr_data['mrparams'], [1, 2, 3, 4])
    assert hdr_data['tr'] == 1
    assert hdr_data['flip_angle'] == 2
    assert hdr_data['te'] == 3
    assert hdr_data['ti'] == 4
    assert hdr_data['fov'] == 0

    hdr['tr'] = 5
    hdr['flip_angle'] = 6
    hdr['te'] = 7
    hdr['ti'] = 8
    assert_array_almost_equal(hdr['mrparams'], [5, 6, 7, 8])
    assert_array_almost_equal(hdr_data['mrparams'], [5, 6, 7, 8])

    hdr_data['tr'] = 9
    hdr_data['flip_angle'] = 10
    hdr_data['te'] = 11
    hdr_data['ti'] = 12
    assert_array_almost_equal(hdr['mrparams'], [9, 10, 11, 12])
    assert_array_almost_equal(hdr_data['mrparams'], [9, 10, 11, 12])


class TestMGHImage(tsi.TestSpatialImage, tsi.MmapImageMixin):
    """ Apply general image tests to MGHImage
    """
    image_class = MGHImage
    can_save = True

    def check_dtypes(self, expected, actual):
        # Some images will want dtypes to be equal including endianness,
        # others may only require the same type
        # MGH requires the actual to be a big endian version of expected
        assert expected.newbyteorder('>') == actual


class TestMGHHeader(_TestLabeledWrapStruct):
    header_class = MGHHeader

    def _set_something_into_hdr(self, hdr):
        hdr['dims'] = [4, 3, 2, 1]

    def get_bad_bb(self):
        return b'\xff' + b'\x00' * self.header_class._hdrdtype.itemsize

    # Update tests to account for big-endian requirement
    def test_general_init(self):
        hdr = self.header_class()
        # binaryblock has length given by header data dtype
        binblock = hdr.binaryblock
        assert len(binblock) == hdr.structarr.dtype.itemsize
        # Endianness will always be big, and cannot be set
        assert hdr.endianness == '>'
        # You can also pass in a check flag, without data this has no
        # effect
        hdr = self.header_class(check=False)

    def test__eq__(self):
        # Test equal and not equal
        hdr1 = self.header_class()
        hdr2 = self.header_class()
        assert hdr1 == hdr2
        self._set_something_into_hdr(hdr1)
        assert hdr1 != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr1 == hdr2
        # REMOVED as_byteswapped() test
        # Check comparing to funny thing says no
        assert hdr1 != None
        assert hdr1 != 1

    def test_to_from_fileobj(self):
        # Successful write using write_to
        hdr = self.header_class()
        str_io = io.BytesIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        assert hdr2.endianness == '>'
        assert hdr2.binaryblock == hdr.binaryblock

    def test_endian_guess(self):
        # Check guesses of endian
        eh = self.header_class()
        assert eh.endianness == '>'
        assert self.header_class.guessed_endian(eh) == '>'

    def test_bytes(self):
        # Test get of bytes
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        # Do a set into the header, and try again.  The specifics of 'setting
        # something' will depend on the nature of the bytes object
        self._set_something_into_hdr(hdr1)
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        # Short binaryblocks give errors (here set through init)
        # Long binaryblocks are truncated
        with pytest.raises(WrapStructError):
            self.header_class(bb[:self.header_class._hdrdtype.itemsize - 1])

        # Checking set to true by default, and prevents nonsense being
        # set into the header.
        bb_bad = self.get_bad_bb()
        if bb_bad is None:
            return
        with imageglobals.LoggingOutputSuppressor():
            with pytest.raises(HeaderDataError):
                self.header_class(bb_bad)

        # now slips past without check
        _ = self.header_class(bb_bad, check=False)

    def test_as_byteswapped(self):
        # Check byte swapping
        hdr = self.header_class()
        assert hdr.endianness == '>'
        # same code just returns a copy
        for endianness in BIG_CODES:
            hdr2 = hdr.as_byteswapped(endianness)
            assert(hdr2 is not hdr)
            assert hdr2 == hdr

        # Different code raises error
        for endianness in (None,) + LITTLE_CODES:
            with pytest.raises(ValueError):
                hdr.as_byteswapped(endianness)
        # Note that contents is not rechecked on swap / copy
        class DC(self.header_class):
            def check_fix(self, *args, **kwargs):
                raise Exception

        # Assumes check=True default
        with pytest.raises(Exception):
            DC(hdr.binaryblock)

        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped('>')

    def test_checks(self):
        # Test header checks
        hdr_t = self.header_class()
        # _dxer just returns the diagnostics as a string
        # Default hdr is OK
        assert self._dxer(hdr_t) == ''
        # Version should be 1
        hdr = hdr_t.copy()
        hdr['version'] = 2
        assert self._dxer(hdr) == 'Unknown MGH format version'
