"""
Unit tests for GridData models
"""

import numpy
import pytest
from astropy.wcs import WCS
from xarray import DataArray

from ska_sdp_datamodels.gridded_visibility.grid_vis_model import (
    ConvolutionFunction,
    GridData,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)

N_CHAN = 100
N_POL = 2
NV = 256
NU = 256
# pylint: disable=duplicate-code
GD_WCS_HEADER = {
    "CTYPE1": "UU",
    "CTYPE2": "VV",
    "CTYPE3": "STOKES",
    "CTYPE4": "FREQ",
    "CRVAL1": 40.0,  # RA in deg
    "CRVAL2": 0.0,  # DEC in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT4": 10.0,  # delta between channel_bandwidth values
}


NW = 4
OVERSAMPLING = 1
SUPPORT = 3
CF_WCS_HEADER = {
    "CTYPE1": "UU",
    "CTYPE2": "VV",
    "CTYPE3": "DUU",
    "CTYPE4": "DVV",
    "CTYPE5": "WW",
    "CTYPE6": "STOKES",
    "CTYPE7": "FREQ",
    "CDELT1": -0.1,
    "CDELT2": 0.1,
    "CDELT3": 1.0,
    "CDELT4": 1.0,
    "CDELT5": 1.0,
    "CDELT6": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT7": 1,  # delta between frequency values
}


@pytest.fixture(scope="module", name="result_grid_data")
def fixture_griddata():
    """
    Generate a simple grid data object using GridData.constructor.
    """
    data = numpy.ones((N_CHAN, N_POL, NV, NU))
    pol_frame = PolarisationFrame("stokesIV")
    grid_wcs = WCS(header=GD_WCS_HEADER, naxis=4)
    grid_data = GridData.constructor(data, pol_frame, grid_wcs)
    return grid_data


@pytest.fixture(scope="module", name="result_convolution_function")
def fixture_convolution_function():
    """
    Generate convolution function object with ConvolutionFunction.constructor
    """
    data = numpy.ones(
        (N_CHAN, N_POL, NW, OVERSAMPLING, OVERSAMPLING, SUPPORT, SUPPORT)
    )
    polarisation_frame = PolarisationFrame("stokesIV")
    cf_wcs = WCS(header=CF_WCS_HEADER, naxis=7)
    convolution_function = ConvolutionFunction.constructor(
        data, cf_wcs, polarisation_frame
    )
    return convolution_function


def test_grid_data_constructor_coords(result_grid_data):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["frequency", "polarisation", "v", "u"]
    result_coords = result_grid_data.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["v"].shape == (NV,)
    assert result_coords["u"].shape == (NU,)
    assert result_coords["v"][0] != result_coords["v"][-1]
    assert result_coords["u"][0] != result_coords["u"][-1]


def test_grid_data_constructor_data_vars(result_grid_data):
    """
    Constructor correctly generates data variables
    """
    result_data_vars = result_grid_data.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()


def test_grid_data_constructor_attrs(result_grid_data):
    """
    Constructor correctly generates attributes
    """
    result_attrs = result_grid_data.attrs

    assert len(result_attrs) == 2
    assert result_attrs["data_model"] == "GridData"
    assert result_attrs["_polarisation_frame"] == "stokesIV"


def test_qa_grid_data(result_grid_data):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    expected_data = {
        "shape": f"({N_CHAN}, {N_POL}, {NV}, {NU})",
        "max": 1.0,
        "min": 1.0,
        "rms": 0.0,
        "sum": 13107200.0,
        "medianabs": 1.0,
        "median": 1.0,
    }
    result_qa = result_grid_data.griddata_acc.qa_grid_data(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"


def test_property_accessor(result_grid_data):
    """
    GridData.griddata_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_grid_data.griddata_acc

    assert accessor_object.nchan == N_CHAN
    assert accessor_object.npol == N_POL
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesIV")
    assert accessor_object.shape == (N_CHAN, N_POL, NV, NU)
    for key, value in GD_WCS_HEADER.items():
        assert (
            accessor_object.griddata_wcs.to_header()[key] == value
        ), f"{key} mismatch"


def test_convolution_function_constructor_coords(result_convolution_function):
    """
    Constructor correctly generates coordinates
    """
    expected_coords_keys = [
        "frequency",
        "polarisation",
        "dv",
        "du",
        "w",
        "v",
        "u",
    ]
    result_coords = result_convolution_function.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["w"].shape == (NW,)


def test_convolution_function_constructor_data_vars(
    result_convolution_function,
):
    """
    Constructor correctly generates data variables
    """

    result_data_vars = result_convolution_function.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()


def test_convolution_function_constructor_attrs(result_convolution_function):
    """
    Constructor correctly generates attributes
    """
    result_attrs = result_convolution_function.attrs

    assert len(result_attrs) == 2
    assert result_attrs["data_model"] == "ConvolutionFunction"
    assert result_attrs["_polarisation_frame"] == "stokesIV"


def test_convolution_function_qa_convolution_function(
    result_convolution_function,
):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    expected_data = {
        "shape": f"({N_CHAN}, {N_POL}, {NW}, {OVERSAMPLING}, {OVERSAMPLING}, "
        f"{SUPPORT}, {SUPPORT})",
        "max": 1.0,
        "min": 1.0,
        "rms": 0.0,
        "sum": 7200.0,
        "medianabs": 1.0,
        "median": 1.0,
    }
    accessor_object = result_convolution_function.convolutionfunction_acc
    result_qa = accessor_object.qa_convolution_function(context="Test")
    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"


def test_convolution_function_property_accessor(result_convolution_function):
    """
    ConvolutionFunction.convolutionfunction_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_convolution_function.convolutionfunction_acc

    assert accessor_object.nchan == N_CHAN
    assert accessor_object.npol == N_POL
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesIV")
    assert accessor_object.shape == (
        N_CHAN,
        N_POL,
        NW,
        OVERSAMPLING,
        OVERSAMPLING,
        SUPPORT,
        SUPPORT,
    )
    for key, value in CF_WCS_HEADER.items():
        assert (
            accessor_object.cf_wcs.to_header()[key] == value
        ), f"{key} mismatch"
