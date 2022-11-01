""" Unit tests for the Sky Model
"""
# pylint: disable=duplicate-code
# make python-format
# make python lint
import numpy
import pytest
from astropy.wcs import WCS

from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent, SkyModel

N_CHAN = 100
N_POL = 1
Y = 512
X = 256
CLEAN_BEAM = {"bmaj": 0.1, "bmin": 0.1, "bpa": 0.1}

WCS_HEADER = {
    "CTYPE1": "RA---SIN",
    "CTYPE2": "DEC--SIN",
    "CTYPE3": "STOKES",  # no units, so no CUNIT3
    "CTYPE4": "FREQ",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CUNIT4": "Hz",
    "CRPIX1": 120,  # CRPIX1-4 are reference pixels
    "CRPIX2": 120,
    "CRPIX3": 1,
    "CRPIX4": 1,
    "CRVAL1": 40.0,  # RA in deg
    "CRVAL2": 0.0,  # DEC in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT4": 10.0,  # delta between frequency values
}

#  Create an Image object

DATA = numpy.ones((N_CHAN, N_POL, Y, X))
POL_FRAME = PolarisationFrame("stokesI")
WCS = WCS(header=WCS_HEADER, naxis=4)
IMAGE = Image.constructor(DATA, POL_FRAME, WCS, clean_beam=CLEAN_BEAM)
COMPONENTS = None

# Create a Configuration and a GainTable object

NAME = "MID"
LOCATION = (5109237.71471275, 2006795.66194638, -3239109.1838011)
NAMES = "M000"
XYZ = 222
MOUNT = "altaz"
FRAME = None
RECEPTOR_FRAME = ReceptorFrame("stokesI")
DIAMETER = 13.5
OFFSET = 0.0
STATIONS = 0
VP_TYPE = "MEERKAT"
CONFIGUTRATION = Configuration.constructor(
    NAME,
    LOCATION,
    NAMES,
    XYZ,
    MOUNT,
    FRAME,
    RECEPTOR_FRAME,
    DIAMETER,
    OFFSET,
    STATIONS,
    VP_TYPE,
)
GAIN = numpy.ones((1, 1, 1, 1, 1))
TIME = numpy.ones(1)
INTERVAL = numpy.ones(1)
WEIGHT = numpy.ones((1, 1, 1, 1, 1))
RESIDUAL = numpy.ones((1, 1, 1, 1))
FREQUENCY = numpy.ones(1)
PHASECENTRE = (180.0, -35.0)
JONES_TYPE = "T"
GAINTABLE = GainTable.constructor(
    GAIN,
    TIME,
    INTERVAL,
    WEIGHT,
    RESIDUAL,
    FREQUENCY,
    RECEPTOR_FRAME,
    PHASECENTRE,
    CONFIGUTRATION,
    JONES_TYPE,
)


# SkyModel Class tests


@pytest.fixture(scope="module", name="result_sky_model")
def fixture_sky_model():
    """
    Generate a sky model object using __init__.
    """
    mask = "Test_mask"
    fixed = True
    sky_model = SkyModel(IMAGE, COMPONENTS, GAINTABLE, mask, fixed)
    return sky_model


def test_sky_model__str__(result_sky_model):
    """
    Check __str__() returns the correct string
    """
    components = ""
    sky_model_text = "SkyModel: fixed: True\n"
    sky_model_text += f"{str(components)}\n"
    sky_model_text += f"{str(IMAGE)}\n"
    sky_model_text += "Test_mask\n"
    sky_model_text += f"{str(GAINTABLE)}"
    assert str(result_sky_model) == sky_model_text


# SkyComponent Class tests


@pytest.fixture(scope="module", name="result_sky_component")
def fixture_sky_component():
    """
    Generate a simple sky component object using __init__.
    """
    direction = (180.0, -35.0)
    frequency = numpy.ones(1)
    name = "test"
    flux = numpy.ones((1, 1))
    shape = "Point"
    polarisation_frame = PolarisationFrame("stokesI")
    sky_component = SkyComponent(
        direction,
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
        params=None,
    )
    return sky_component


def test_sky_component_nchan(result_sky_component):
    """
    Check nchans returns correct data
    """
    nchans = result_sky_component.nchan
    assert nchans == 1


def test_sky_component_npol(result_sky_component):
    """
    Check npols returns correct data
    """
    npols = result_sky_component.npol
    assert npols == 1


def test_sky_component__str__(result_sky_component):
    """
    Check __str__() returns the correct string
    """
    params = {}
    sky_comp_text = "SkyComponent:\n"
    sky_comp_text += "\tName: test\n"
    sky_comp_text += "\tFlux: [[1.]]\n"
    sky_comp_text += "\tFrequency: [1.]\n"
    sky_comp_text += "\tDirection: (180.0, -35.0)\n"
    sky_comp_text += "\tShape: Point\n"
    sky_comp_text += f"\tParams: {params}\n"
    sky_comp_text += "\tPolarisation frame: stokesI\n"
    assert str(result_sky_component) == sky_comp_text