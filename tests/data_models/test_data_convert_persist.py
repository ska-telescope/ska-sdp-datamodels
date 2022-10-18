# pylint: disable=attribute-defined-outside-init,invalid-name,
# pylint: disable=missing-function-docstring
# pylint: disable=no-name-in-module,import-error

"""
Unit tests for functions in data_convert_persist.
The functions facilitate persistence of data models using HDF5
"""

import logging
import unittest

import astropy.units as u
import numpy
import xarray
from astropy.coordinates import SkyCoord

from src.processing_components.calibration.operations import (
    create_gaintable_from_visibility,
)
from src.processing_components.calibration.pointing import (
    create_pointingtable_from_visibility,
)
from src.processing_components.flagging.base import (
    create_flagtable_from_visibility,
)
from src.processing_components.griddata import (
    create_convolutionfunction_from_image,
)
from src.processing_components.griddata.operations import (
    create_griddata_from_image,
)
from src.processing_components.image import create_image
from src.processing_components.imaging import dft_skycomponent_visibility
from src.processing_components.simulation import (
    create_named_configuration,
    simulate_gaintable,
)
from src.processing_components.simulation.pointing import (
    simulate_pointingtable,
)
from src.processing_components.visibility.base import create_visibility
from src.ska_sdp_datamodels.data_convert_persist import (
    export_convolutionfunction_to_hdf5,
    export_flagtable_to_hdf5,
    export_gaintable_to_hdf5,
    export_griddata_to_hdf5,
    export_image_to_hdf5,
    export_pointingtable_to_hdf5,
    export_skycomponent_to_hdf5,
    export_skymodel_to_hdf5,
    export_visibility_to_hdf5,
    import_convolutionfunction_from_hdf5,
    import_flagtable_from_hdf5,
    import_gaintable_from_hdf5,
    import_griddata_from_hdf5,
    import_image_from_hdf5,
    import_pointingtable_from_hdf5,
    import_skycomponent_from_hdf5,
    import_skymodel_from_hdf5,
    import_visibility_from_hdf5,
)
from src.ska_sdp_datamodels.memory_data_models import SkyComponent, SkyModel
from src.ska_sdp_datamodels.polarisation_data_models import PolarisationFrame

log = logging.getLogger("src-logger")

log.setLevel(logging.INFO)


class TestDataModelHelpers(unittest.TestCase):
    def _data_model_equals(self, ds_new, ds_ref):
        """Check if two xarray objects are identical except to values

        Precision in lost in HDF files at close to the machine
        precision so we cannot reliably use xarray.equals().
        So this function is specific to this set of tests

        Throws AssertionError or returns True

        :param ds_ref: xarray Dataset or DataArray
        :param ds_new: xarray Dataset or DataArray
        :return: True or False
        """
        for coord in ds_ref.coords:
            assert coord in ds_new.coords
        for coord in ds_new.coords:
            assert coord in ds_ref.coords
        for var in ds_ref.data_vars:
            assert var in ds_new.data_vars
        for var in ds_new.data_vars:
            assert var in ds_ref.data_vars
        for attr in ds_ref.attrs.keys():
            assert attr in ds_new.attrs.keys()
        for attr in ds_new.attrs.keys():
            assert attr in ds_ref.attrs.keys()

        return True

    def setUp(self):
        from src.processing_components.parameters import rascil_path

        self.results_dir = rascil_path("test_results")

        self.mid = create_named_configuration("MID", rmax=1000.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component
        # is specified relative (for now).
        # This means that the component should end up at
        # the position phasecentre+compredirection
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.compabsdirection = SkyCoord(
            ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.comp = SkyComponent(
            direction=self.compabsdirection,
            frequency=self.frequency,
            flux=self.flux,
        )

        self.verbose = False

    def test_readwritevisibility(self):
        self.vis = create_visibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)
        if self.verbose:
            print(self.vis)
            print(self.vis.configuration)
        export_visibility_to_hdf5(
            self.vis,
            f"{self.results_dir}/test_data_convert_persist_visibility.hdf",
        )
        newvis = import_visibility_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_visibility.hdf"
        )
        assert self._data_model_equals(newvis, self.vis)

    def test_readwritegaintable(self):
        self.vis = create_visibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        gt = create_gaintable_from_visibility(
            self.vis, timeslice="auto", jones_type="G"
        )
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)
        if self.verbose:
            print(gt)
        export_gaintable_to_hdf5(
            gt, f"{self.results_dir}/test_data_convert_persist_gaintable.hdf"
        )
        newgt = import_gaintable_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_gaintable.hdf"
        )
        assert self._data_model_equals(newgt, gt)

    def test_readwriteflagtable(self):
        self.vis = create_visibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        ft = create_flagtable_from_visibility(self.vis, timeslice="auto")
        if self.verbose:
            print(ft)
        export_flagtable_to_hdf5(
            ft, f"{self.results_dir}/test_data_convert_persist_flagtable.hdf"
        )
        newft = import_flagtable_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_flagtable.hdf"
        )
        assert self._data_model_equals(newft, ft)

    def test_readwritepointingtable(self):
        self.vis = create_visibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        pt = create_pointingtable_from_visibility(self.vis, timeslice="auto")
        pt = simulate_pointingtable(pt, pointing_error=0.001)
        if self.verbose:
            print(pt)
        export_pointingtable_to_hdf5(
            pt,
            f"{self.results_dir}/test_data_convert_persist_pointingtable.hdf",
        )
        newpt = import_pointingtable_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_pointingtable.hdf"
        )
        assert self._data_model_equals(newpt, pt)

    def test_readwriteimage(self):
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        im["pixels"].data[...] = 1.0
        export_image_to_hdf5(
            im, f"{self.results_dir}/test_data_convert_persist_image.hdf"
        )
        newim = import_image_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_image.hdf"
        )
        assert self._data_model_equals(newim, im)

    def test_readwriteimage_zarr(self):
        """
        Test to see if an image can be written to
        and read from a zarr file
        """
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        rand = numpy.random.random(im["pixels"].shape)
        im["pixels"].data[...] = rand
        if self.verbose:
            print(im)
        import os

        # We cannot save dicts to a netcdf file
        im.attrs["clean_beam"] = ""

        store = os.path.expanduser(
            f"{self.results_dir}/test_data_convert_persist_image.zarr"
        )
        im.to_zarr(
            store=store,
            chunk_store=store,
            mode="w",
        )
        del im
        newim = xarray.open_zarr(store, chunk_store=store)
        assert newim["pixels"].data.compute().all() == rand.all()

    def test_readwriteskycomponent(self):
        export_skycomponent_to_hdf5(
            self.comp,
            f"{self.results_dir}/test_data_convert_persist_skycomponent.hdf",
        )
        newsc = import_skycomponent_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_skycomponent.hdf"
        )

        assert newsc.flux.shape == self.comp.flux.shape
        assert numpy.max(numpy.abs(newsc.flux - self.comp.flux)) < 1e-15

    def test_readwriteskymodel(self):
        self.vis = create_visibility(
            self.mid,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            polarisation_frame=PolarisationFrame("linear"),
            weight=1.0,
        )
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        gt = create_gaintable_from_visibility(self.vis, timeslice="auto")
        sm = SkyModel(components=[self.comp], image=im, gaintable=gt)
        export_skymodel_to_hdf5(
            sm, f"{self.results_dir}/test_data_convert_persist_skymodel.hdf"
        )
        newsm = import_skymodel_from_hdf5(
            f"{self.results_dir}/test_data_convert_persist_skymodel.hdf"
        )

        assert newsm.components[0].flux.shape == self.comp.flux.shape

    def test_readwritegriddata(self):
        # This fails on comparison of the v axis.
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        gd = create_griddata_from_image(im)
        export_griddata_to_hdf5(
            gd, "%s/test_data_convert_persist_griddata.hdf" % self.results_dir
        )
        newgd = import_griddata_from_hdf5(
            "%s/test_data_convert_persist_griddata.hdf" % self.results_dir
        )
        assert self._data_model_equals(newgd, gd)

    def test_readwriteconvolutionfunction(self):
        # This fails on comparison of the v axis.
        im = create_image(
            phasecentre=self.phasecentre,
            frequency=self.frequency,
            npixel=256,
            polarisation_frame=PolarisationFrame("stokesIQUV"),
        )
        cf = create_convolutionfunction_from_image(im)
        if self.verbose:
            print(cf)
        export_convolutionfunction_to_hdf5(
            cf,
            f"{self.results_dir}/"
            f"test_data_convert_persist_convolutionfunction.hdf",
        )
        newcf = import_convolutionfunction_from_hdf5(
            f"{self.results_dir}/"
            f"test_data_convert_persist_convolutionfunction.hdf"
        )

        assert self._data_model_equals(newcf, cf)


if __name__ == "__main__":
    unittest.main()
