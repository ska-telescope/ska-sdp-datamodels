# pylint: disable=inconsistent-return-statements, too-few-public-methods
"""
Unit Tests to create GainTable
from CASA Tables
"""
from unittest.mock import patch

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.units import Quantity

from ska_sdp_datamodels.calibration.calibration_functions import (
    _generate_configuration_from_cal_table,
    _get_phase_centre_from_cal_table,
    import_gaintable_from_casa_cal_table,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.science_data_model import ReceptorFrame

NTIMES = 4
NANTS = 6
NFREQ = 3
TEL_NAME = "MY-SKA"


class MockBaseTable:
    """
    Mock Base Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TIME":
            return numpy.array(
                [4.35089331e09, 4.35089332e09, 4.35089333e09, 4.35089334e09]
            )

        if columnname == "INTERVAL":
            return numpy.repeat(10.0, NTIMES)

        if columnname == "CPARAM":
            return numpy.ones((NTIMES, NANTS, NFREQ, 2))

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0, 1])

    def getkeyword(self, keyword=None):
        """
        Get the value of a table keyword
        """
        if keyword == "VisCal":
            return "B Jones"


class MockFullBaseTable:
    """
    Mock Base Table Class with a row for each time and antenna combination
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TIME":
            return numpy.concatenate(
                (
                    numpy.repeat(4.35089331e09, NANTS),
                    numpy.repeat(4.35089332e09, NANTS),
                    numpy.repeat(4.35089333e09, NANTS),
                    numpy.repeat(4.35089334e09, NANTS),
                )
            )

        if columnname == "INTERVAL":
            return numpy.repeat(10.0, NTIMES * NANTS)

        if columnname == "CPARAM":
            return numpy.ones((NTIMES * NANTS, NFREQ, 2))

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0, 1])

    def getkeyword(self, keyword=None):
        """
        Get the value of a table keyword
        """
        if keyword == "VisCal":
            return "B Jones"


class MockLeakageBaseTable:
    """
    Mock Base Table Class with leakage terms rather than gains
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TIME":
            return numpy.concatenate(
                (
                    numpy.repeat(4.35089331e09, NANTS),
                    numpy.repeat(4.35089332e09, NANTS),
                    numpy.repeat(4.35089333e09, NANTS),
                    numpy.repeat(4.35089334e09, NANTS),
                )
            )

        if columnname == "INTERVAL":
            return numpy.repeat(10.0, NTIMES * NANTS)

        if columnname == "CPARAM":
            return 0.1 * numpy.ones((NTIMES * NANTS, NFREQ, 2))

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0, 1])

    def getkeyword(self, keyword=None):
        """
        Get the value of a table keyword
        """
        if keyword == "VisCal":
            return "Df Jones"


class MockDelayBaseTable:
    """
    Mock Base Table Class with leakage terms rather than gains
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TIME":
            return numpy.concatenate(
                (
                    numpy.repeat(4.35089331e09, NANTS),
                    numpy.repeat(4.35089332e09, NANTS),
                    numpy.repeat(4.35089333e09, NANTS),
                    numpy.repeat(4.35089334e09, NANTS),
                )
            )

        if columnname == "INTERVAL":
            return numpy.repeat(10.0, NTIMES * NANTS)

        if columnname == "FPARAM":
            return numpy.ones((NTIMES * NANTS, 1, 2))

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0])

    def getkeyword(self, keyword=None):
        """
        Get the value of a table keyword
        """
        if keyword == "VisCal":
            return "K Jones"


class MockKcrossBaseTable:
    """
    Mock Base Table Class with leakage terms rather than gains
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TIME":
            return numpy.concatenate(
                (
                    numpy.repeat(4.35089331e09, NANTS),
                    numpy.repeat(4.35089332e09, NANTS),
                    numpy.repeat(4.35089333e09, NANTS),
                    numpy.repeat(4.35089334e09, NANTS),
                )
            )

        if columnname == "INTERVAL":
            return numpy.repeat(10.0, NTIMES * NANTS)

        if columnname == "FPARAM":
            delay = numpy.zeros((NTIMES * NANTS, 1, 2))
            delay[..., 0] = 1
            return delay

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0])

    def getkeyword(self, keyword=None):
        """
        Get the value of a table keyword
        """
        if keyword == "VisCal":
            return "Kcross Jones"


class MockSpectralWindowTable:
    """
    Mock Spectral Window Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "CHAN_FREQ":
            return numpy.array([[8.0e9, 8.1e9, 8.2e9], [8.4e9, 8.5e9, 8.6e9]])

        if columnname == "NUM_CHAN":
            return numpy.array([NFREQ, NFREQ])


class MockDelaySpectralWindowTable:
    """
    Mock Spectral Window Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "CHAN_FREQ":
            return numpy.array([[8.1e9]])

        if columnname == "NUM_CHAN":
            return numpy.array([1])


class MockAntennaTable:
    """
    Mock Antenna Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name from MS File
        """
        if columnname == "NAME":
            return ["ANT1", "ANT2", "ANT3", "ANT4", "ANT5", "ANT6"]

        if columnname == "MOUNT":
            return ["ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ"]

        if columnname == "DISH_DIAMETER":
            return numpy.array([25.0, 25.0, 25.0, 25.0, 25.0, 25.0])

        if columnname == "POSITION":
            return numpy.array(
                [
                    [-1601162.0, -5042003.0, 3554915.0],
                    [-1601192.0190292, -5042007.78341262, 3554960.73493029],
                    [-1601147.19047704, -5042040.12425644, 3554894.80919799],
                    [-1601110.11175873, -5041807.16312437, 3554839.91628013],
                    [-1601405.58491341, -5042041.04214758, 3555275.06577525],
                    [-1601093.35329757, -5042182.23690452, 3554815.49897078],
                ]
            )

        if columnname == "OFFSET":
            return numpy.array(
                [
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                    [2.18848512e-03, 0.00000000e00, 0.00000000e00],
                    [-3.02790361e-03, 0.00000000e00, 0.00000000e00],
                    [-9.89315100e-04, 0.00000000e00, 0.00000000e00],
                    [5.09647129e-04, 0.00000000e00, 0.00000000e00],
                    [-2.87800725e-03, 0.00000000e00, 0.00000000e00],
                ]
            )

        if columnname == "STATION":
            return [
                "SKAMID-CORE",
                "SKAMID-CORE",
                "SKAMID-CORE",
                "SKAMID-ARM1",
                "SKAMID-ARM2",
                "SKAMID-ARM3",
            ]


class MockFieldTable:
    """
    Mock Field Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name from a table
        """
        if columnname == "PHASE_DIR":
            return numpy.array([[[0.0, 0.0]]])


class MockObservationTable:
    """
    Mock Observation Table Class
    """

    def getcol(self, columnname=None):
        """
        Get column name
        """
        if columnname == "TELESCOPE_NAME":
            return TEL_NAME


def test_generate_configuration_from_cal_table():
    """
    Test generating configuration from cal table
    """
    result = _generate_configuration_from_cal_table(
        MockAntennaTable(),
        TEL_NAME,
        ReceptorFrame("linear"),
    )

    location = EarthLocation(
        x=Quantity(-1601162.0, "m"),
        y=Quantity(-5042003.0, "m"),
        z=Quantity(3554915.0, "m"),
    )
    assert result.attrs["name"] == TEL_NAME
    assert result.attrs["location"] == location
    assert result.attrs["receptor_frame"] == ReceptorFrame("linear")
    assert result.coords["id"].data.shape == (6,)


def test_get_phase_centre_from_cal_table():
    """
    Test get phase centre from cal table
    """
    result = _get_phase_centre_from_cal_table(MockFieldTable())
    expected = SkyCoord(
        ra=0.0 * u.rad,
        dec=0.0 * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    assert result == expected

