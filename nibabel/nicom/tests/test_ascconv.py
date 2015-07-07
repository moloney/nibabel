""" Testing Siemens "ASCCONV" parser
"""

from os.path import join as pjoin, dirname

from .. import ascconv

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal

DATA_PATH = pjoin(dirname(__file__), 'data')
ASCCONV_INPUT = pjoin(DATA_PATH, 'ascconv_sample.txt')


def test_ascconv_parse():
    with open(ASCCONV_INPUT, 'rt') as fobj:
        contents = fobj.read()
    ascconv_dict = ascconv.parse_ascconv('MrPhoenixProtocol', contents)
    assert_equal(len(ascconv_dict), 917)
    assert_equal(ascconv_dict['tProtocolName'], 'CBU+AF8-DTI+AF8-64D+AF8-1A')
    assert_equal(ascconv_dict['ucScanRegionPosValid'], 1)
    assert_array_almost_equal(ascconv_dict['sProtConsistencyInfo.flNominalB0'],
                              2.89362)
    assert_equal(ascconv_dict['sProtConsistencyInfo.flGMax'], 26)