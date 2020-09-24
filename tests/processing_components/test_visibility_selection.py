"""Unit tests for visibility selectors

"""
import numpy
import unittest

from astropy.coordinates import SkyCoord
import astropy.units as u
from rascil.data_models import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility import blockvisibility_select, blockvisibility_groupby, \
    blockvisibility_groupby_bins, blockvisibility_where
from rascil.processing_components.visibility.base import create_blockvisibility

import logging
log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestVisibilitySelectors(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")
    
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    def test_blockvisibility_groupby_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        times = numpy.array([result[0] for result in blockvisibility_groupby(bvis, "time")])
        assert times.all() == bvis.time.all()

    def test_blockvisibility_groupby_bins_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        for result in blockvisibility_groupby_bins(bvis, "time", 3):
            print(result[0])

    def test_blockvisibility_where(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        print(blockvisibility_where(bvis, bvis.data["flags"] == 0))

    def test_blockvisibility_where_absu(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        print(blockvisibility_where(bvis, numpy.abs(bvis.u_lambda) < 30.0))


    def test_blockvisibility_select_time(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        times = bvis.time
        selected_bvis = blockvisibility_select(bvis, {"time": slice(times[1], times[2])})
        assert len(selected_bvis.time) == 2

    def test_blockvisibility_select_frequency(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        frequency = bvis.frequency
        selected_bvis = blockvisibility_select(bvis, {"frequency": slice(frequency[1], frequency[2])})
        assert len(selected_bvis.frequency) == 2

