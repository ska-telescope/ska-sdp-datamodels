# pylint: disable=too-many-ancestors,too-many-arguments,too-many-locals
# pylint: disable=invalid-name, unexpected-keyword-arg

"""
Visibility data model.
"""

import warnings
from typing import Optional, Union

import numpy
import pandas
import xarray
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy.typing import NDArray

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    QualityAssessment,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin


class Visibility(xarray.Dataset):
    """
    Container for visibilities associated to an observation with one direction.
    It only stores *rectangular* data, i.e. with consistent baseline, frequency
    and polarisation axes in time (unlike Measurement Sets which are much
    more flexible).

    **Coordinates**

    - time: centre times of visibility samples, in seconds elapsed since the
      MJD reference epoch (on the UTC scale), ``[ntimes]``.

    - baselines: pandas.MultiIndex holding integer tuples that represent
      pairs (antenna1, antenna2). Autocorrelation baselines are included.

    - frequency: centre frequencies of channels in Hz, ``[nchan]``.

    - polarisation: string labels for the polarisation axis, ``[npol]``.
      Matches the ``polarisation_frame`` attribute (see below). There are
      multiple possibilities here. For example, if
      ``["XX", "XY", "YX", "YY"]``, ``vis`` represents unrolled visibility
      matrices expressed in a linear basis. If ``["I", "Q", "U", "V"]``,
      ``vis`` represents stokes vectors. May also be just ``["I"]`` for pure
      stokes I visibilities, etc.

    - spatial: string labels for the columns of the ``uvw`` data variable.
      This is always the 3-element sequence ``["u", "v", "w"]``.

    **Data Variables**

    - vis: visibility data, complex-valued
      ``[ntimes, nbaselines, nchan, npol]``.

    - uvw: (u, v, w) coordinates expressed in metres,
      ``[ntimes, nbaselines, 3]``. This departs from the classical
      convention of expressing (u, v, w) in wavelengths, but reduces memory
      footprint by a factor ``nchan``.
      NOTE: the sign convention is the *opposite* of what Measurement Set uses.
      https://casa.nrao.edu/casadocs/casa-5.4.1/reference-material/measurement-set

    - weight: weights are the inverse noise variances associated with each
      data point; real-valued, same shape as ``vis``.

    - flags: flags associated with the data, integer-valued, same shape as
      ``vis``. 0 means valid data, 1 means should be ignored.

    - datetime: centre times of visibility samples, in np.datetime64 format,
      ``[ntimes]``. Effectively a copy of the "time" coordinate but with a
      different representation.

    - integration_time: integration times in seconds, ``[ntimes]``.

    - channel_bandwidth: channel widths in Hz, ``[nchan]``.

    **Attributes**

    - phasecentre: phase centre coordinates as an astropy SkyCoord object.

    - configuration: Configuration object describing the array with which the
      visibilities were observed.

    - _polarisation_frame: PolarisationFrame object describing the
      polarisation representation of the visibility data.

    - source: source name as a string

    - meta: either None or an optional user-defined dictionary of additional
      metadata.

    - data_model: name of this class, used internally for saving to / loading
      from files.

    Here is an example::

        <xarray.Visibility>
        Dimensions:            (baselines: 6670, frequency: 3, polarisation: 4, time: 3, spatial: 3)
        Coordinates:
          * time               (time) float64 5.085e+09 5.085e+09 5.085e+09
          * baselines          (baselines) MultiIndex
          - antenna1           (baselines) int64 0 0 0 0 0 0 ... 112 112 112 113 113 114
          - antenna2           (baselines) int64 0 1 2 3 4 5 ... 112 113 114 113 114 114
          * frequency          (frequency) float64 1e+08 1.05e+08 1.1e+08
          * polarisation       (polarisation) <U2 'XX' 'XY' 'YX' 'YY'
          * spatial            (spatial) <U1 'u' 'v' 'w'
        Data variables:
            integration_time   (time) float32 99.72697 99.72697 99.72697
            datetime           (time) datetime64[ns] 2000-01-01T03:54:07.843184299 .....
            vis                (time, baselines, frequency, polarisation) complex128 ...
            weight             (time, baselines, frequency, polarisation) float32 0.0...
            flags              (time, baselines, frequency, polarisation) int32 0.0...
            uvw                (time, baselines, spatial) float64 0.0 0.0 ... 0.0 0.0
            channel_bandwidth  (frequency) float64 1e+07 1e+07 1e+07
        Attributes:
            data_model:          Visibility
            phasecentre:         <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:       <xarray.Configuration>Dimensions:   (id: 115, spat...
            _polarisation_frame: linear
            source:              unknown
            meta:                None
    """  # noqa:E501 pylint: disable=line-too-long

    __slots__ = ("_imaging_weight",)

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)
        self._imaging_weight = None

    @classmethod
    def constructor(
        cls,
        frequency: Optional[NDArray] = None,
        channel_bandwidth: Optional[NDArray] = None,
        phasecentre: Optional[SkyCoord] = None,
        configuration: Optional[Configuration] = None,
        uvw: Optional[NDArray] = None,
        time: Optional[NDArray] = None,
        vis: Optional[NDArray] = None,
        weight: Optional[NDArray] = None,
        integration_time: Optional[NDArray] = None,
        flags: Optional[NDArray] = None,
        baselines: Optional[pandas.MultiIndex] = None,
        polarisation_frame: PolarisationFrame = PolarisationFrame("stokesI"),
        source: str = "anonymous",
        meta: Optional[dict] = None,
        low_precision: Union[str, type] = "float64",
    ):
        """
        Create a new Visibility instance.

        :param frequency: Centre frequencies of channels in Hz [nchan]
        :type frequency: ndarray or None, optional

        :param channel_bandwidth: Channel bandwidths in Hz [nchan]
        :type channel_bandwidth: ndarray or None, optional

        :param phasecentre: Coordinates of the phase centre
        :type phasecentre: astropy.coordinates.SkyCoord or None, optional

        :param configuration: Configuration object describing the array with
            which the visibilities were observed.
        :type configuration: Configuration or None, optional

        :param uvw: UVW coordinates in metres [ntimes, nbaselines, 3]. The sign
            convention is the *opposite* of that of Measurement Set.
        :type uvw: ndarray or None, optional

        :param time: Centre times of visibility samples, in seconds elapsed
            since the MJD reference epoch (on the UTC scale) [ntimes]
        :type time: ndarray or None, optional

        :param vis: Visibility data [ntimes, nbaselines, nchan, npol].
        :type vis: ndarray or None, optional

        :param baselines: Sequence of baselines as a pandas.MultiIndex object;
            it is expected to contain two levels (in pandas parlance) called
            "antenna1" and "antenna2", in that order.
        :type baselines: pandas.MultiIndex

        :param flags: Flags associated with the visibility data,
            integer-valued, same shape as ``vis`` argument.
        :type flags: ndarray or None, optional

        :param weight: Weights of the visibility data, i.e. inverse of noise
            variances for each data point, same shape as ``vis`` argument.
        :type weight: ndarray or None, optional

        :param integration_time: Integration times in seconds [ntimes].
        :type integration_time: ndarray or None, optional

        :param polarisation_frame: PolarisationFrame object describing the
            polarisation representation of the visibility data.
        :type polarisation_frame: PolarisationFrame

        :param source: Source name.
        :type source: str, optional

        :param meta: Optional dictionary of user-defined metadata to carry.
        :type meta: dict or None, optional

        :param low_precision: numpy dtype under which to store the
            integration_time and weight data variables. Can be given as string
            or numpy dtype (e.g. "float64" or np.float64).
        :type low_precision: str or type, optional
        """
        if weight is None:
            weight = numpy.ones(vis.shape)
        else:
            assert weight.shape == vis.shape

        if integration_time is None:
            integration_time = numpy.ones_like(time)
        else:
            assert len(integration_time) == len(time)

        # Define the names of the dimensions
        coords = {  # pylint: disable=duplicate-code
            "time": time,
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "spatial": ["u", "v", "w"],
        }

        datavars = {}
        datavars["integration_time"] = xarray.DataArray(
            integration_time.astype(low_precision),
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["vis"] = xarray.DataArray(
            vis,
            dims=["time", "baselines", "frequency", "polarisation"],
            attrs={"units": "Jy"},
        )
        datavars["weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["flags"] = xarray.DataArray(
            flags.astype(int),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["uvw"] = xarray.DataArray(
            uvw, dims=["time", "baselines", "spatial"], attrs={"units": "m"}
        )

        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"}
        )

        attrs = {}
        attrs["data_model"] = "Visibility"
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["source"] = source
        attrs["phasecentre"] = phasecentre
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["meta"] = meta

        return cls(datavars, coords=coords, attrs=attrs)

    @property
    def imaging_weight(self):
        """
        Legacy data attribute. Deprecated.
        """
        warnings.warn(
            "imaging_weight is deprecated, please use weight instead",
            DeprecationWarning,
        )
        if self._imaging_weight is None:
            self._imaging_weight = xarray.DataArray(
                self.weight.data.astype(self.weight.data.dtype),
                dims=["time", "baselines", "frequency", "polarisation"],
            )
        return self._imaging_weight

    @imaging_weight.setter
    def imaging_weight(self, new_img_weight):
        warnings.warn(
            "imaging_weight is deprecated, please use weight instead",
            DeprecationWarning,
        )
        if not new_img_weight.shape == self.weight.data.shape:
            raise ValueError(
                "New imaging weight does not match shape of weight"
            )

        self._imaging_weight = xarray.DataArray(
            new_img_weight.astype(self.weight.data.dtype),
            dims=["time", "baselines", "frequency", "polarisation"],
        )

    def __sizeof__(self):
        """Override default method to return size of dataset
        :return: int
        """
        # Dask uses sizeof() class to get memory occupied by various data
        # objects. For custom data objects like this one, dask falls back to
        # sys.getsizeof() function to get memory usage. sys.getsizeof() in
        # turns calls __sizeof__() magic method to get memory size. Here we
        # override the default method (which gives size of reference table)
        # to return size of Dataset.
        return int(self.nbytes)

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy Visibility

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set visibility data to zero in copied object
        """
        new_vis = super().copy(deep=deep, data=data)
        if zero:
            new_vis["vis"].data[...] = 0.0

        setattr(new_vis, "_imaging_weight", self._imaging_weight)
        return new_vis

    def groupby(
        self, group, squeeze: bool = True, restore_coord_dims: bool = None
    ):
        """Override default method to group _imaging_weight"""
        grouped_dataset = super().groupby(
            group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
        )

        if self._imaging_weight is not None:
            group_imaging_weight = self._imaging_weight.groupby(
                group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
            )

            for (dimension, vis_slice), (_, imaging_weight_slice) in zip(
                grouped_dataset, group_imaging_weight
            ):
                setattr(vis_slice, "_imaging_weight", imaging_weight_slice)
                yield dimension, vis_slice
        else:
            for dimension, vis_slice in grouped_dataset:
                setattr(vis_slice, "_imaging_weight", None)
                yield dimension, vis_slice

    def groupbybins(
        self,
        group,
        bins,
        right=True,
        labels=None,
        precision=3,
        include_lowest=False,
        squeeze=True,
        restore_coord_dims=False,
    ):
        """
        Overwriting groupbybins method.
        See docstring of Dataset.groupbybins
        """
        grouped_dataset = super().groupby_bins(
            group,
            bins,
            right=right,
            labels=labels,
            precision=precision,
            include_lowest=include_lowest,
            squeeze=squeeze,
            restore_coord_dims=restore_coord_dims,
        )

        if self._imaging_weight is not None:
            group_imaging_weight = self._imaging_weight.groupby_bins(
                group,
                squeeze=squeeze,
                bins=bins,
                restore_coord_dims=restore_coord_dims,
                cut_kwargs={
                    "right": right,
                    "labels": labels,
                    "precision": precision,
                    "include_lowest": include_lowest,
                },
            )

            for (dimension, vis_slice), (_, imaging_weight_slice) in zip(
                grouped_dataset, group_imaging_weight
            ):
                setattr(vis_slice, "_imaging_weight", imaging_weight_slice)
                yield dimension, vis_slice
        else:
            for dimension, vis_slice in grouped_dataset:
                setattr(vis_slice, "_imaging_weight", None)
                yield dimension, vis_slice


@xarray.register_dataset_accessor("visibility_acc")
class VisibilityAccessor(XarrayAccessorMixin):
    """
    Visibility property accessor
    """

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self._uvw_lambda = None

    @property
    def rows(self):
        """Rows"""
        return range(len(self._obj.time))

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return len(self._obj["time"])

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return len(self._obj.polarisation)

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self._obj.configuration.configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self._obj["baselines"])

    @property
    def uvw_lambda(self):
        """
        Calculate and set uvw_lambda
        dims=[ntimes, nbaselines, nchan, spatial(3)]
        Note: We omit the frequency and polarisation
            dependency of uvw for the calculation
        """
        if self._uvw_lambda is None:
            k = (
                self._obj["frequency"].data
                / const.c  # pylint: disable=no-member
            ).value
            uvw = self._obj["uvw"].data
            if self.nchan == 1:
                self._uvw_lambda = (uvw * k)[..., numpy.newaxis, :]
            else:
                self._uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)

        return self._uvw_lambda

    @uvw_lambda.setter
    def uvw_lambda(self, new_value):
        """
        Re-set uvw_lambda to a given value if it has been recalculated
        """

        if not new_value.shape == (
            self.ntimes,
            self.nbaselines,
            self.nchan,
            3,
        ):
            raise ValueError(
                "Data shape of new uvw_lambda "
                "incompatible with visibility setup"
            )

        self._uvw_lambda = new_value

    @property
    def u(self):
        """u coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 0]

    @property
    def v(self):
        """v coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 1]

    @property
    def w(self):
        """w coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 2]

    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["vis"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_weight(self):
        """Weight [: npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["weight"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_imaging_weight(self):
        """Flagged Imaging_weight[nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        warnings.warn(
            "flagged_imaging_weight is deprecated, "
            "please use flagged_weight instead",
            DeprecationWarning,
        )
        return self._obj.imaging_weight.data * (1 - self._obj["flags"].data)

    @property
    def nvis(self):
        """Number of visibilities (in total)"""
        return numpy.product(self._obj.vis.shape)

    def qa_visibility(self, context=None) -> QualityAssessment:
        """Assess the quality of Visibility"""

        avis = numpy.abs(self._obj["vis"].data)
        data = {
            "maxabs": numpy.max(avis),
            "minabs": numpy.min(avis),
            "rms": numpy.std(avis),
            "medianabs": numpy.median(avis),
        }
        qa = QualityAssessment(
            origin="qa_visibility", data=data, context=context
        )
        return qa

    def performance_visibility(self):
        """Get info about the visibility

        This works on a single visibility because we
        probably want to send this function to
        the cluster instead of bringing the data back
        :return: bvis info as a dictionary
        """
        bv_info = {
            "number_times": self.ntimes,
            "number_baselines": len(self._obj.baselines),
            "nchan": self.nchan,
            "npol": self.npol,
            "polarisation_frame": self.polarisation_frame.type,
            "nvis": self.ntimes * self.nbaselines * self.nchan * self.npol,
            "size": self._obj.nbytes,
        }
        return bv_info

    def select_uv_range(self, uvmin=0.0, uvmax=1.0e15):
        """Visibility selection functions

        To select by row number::
            selected_bvis = bvis.isel({"time": slice(5, 7)})
        To select by frequency channel::
            selected_bvis = bvis.isel({"frequency": slice(1, 3)})
        To select by frequency::
            selected_bvis = bvis.sel({"frequency": slice(0.9e8, 1.2e8)})
        To select by frequency and polarisation::
            selected_bvis = bvis.sel(
              {"frequency": slice(0.9e8, 1.2e8), "polarisation": ["XX", "YY"]}
            ).dropna(dim="frequency", how="all")

        Select uv range: flag in-place all visibility data
        outside uvrange uvmin, uvmax (wavelengths)
        The flags are set to 1 for all data outside the specified uvrange

        :param uvmin: Minimum uv to flag
        :param uvmax: Maximum uv to flag
        :return: Visibility (with flags applied)
        """
        uvdist_lambda = numpy.hypot(
            self.uvw_lambda[..., 0],
            self.uvw_lambda[..., 1],
        )
        if uvmax is not None:
            self._obj["flags"].data[numpy.where(uvdist_lambda >= uvmax)] = 1
        if uvmin is not None:
            self._obj["flags"].data[numpy.where(uvdist_lambda <= uvmin)] = 1

    def select_r_range(self, rmin=None, rmax=None):
        """
        Select a visibility with stations in a range
        of distance from the array centre
        r is the distance from the array centre in metres

        :param rmax: Maximum r
        :param rmin: Minimum r
        :return: Selected Visibility
        """
        if rmin is None and rmax is None:
            return self._obj

        # Calculate radius from array centre (in 3D)
        # and set it as a data variable
        xyz0 = self._obj.configuration.xyz - self._obj.configuration.xyz.mean(
            "id"
        )
        r = numpy.sqrt(xarray.dot(xyz0, xyz0, dims="spatial"))
        config = self._obj.configuration.assign(radius=r)
        # Now use it for selection
        if rmax is None:
            sub_config = config.where(config["radius"] > rmin, drop=True)
        elif rmin is None:
            sub_config = config.where(config["radius"] < rmax, drop=True)
        else:
            sub_config = config.where(
                config["radius"] > rmin, drop=True
            ).where(config["radius"] < rmax, drop=True)

        ids = list(sub_config.id.data)
        baselines = self._obj.baselines.where(
            self._obj.baselines.antenna1.isin(ids), drop=True
        ).where(self._obj.baselines.antenna2.isin(ids), drop=True)
        sub_bvis = self._obj.sel({"baselines": baselines}, drop=True)
        setattr(
            sub_bvis,
            "_imaging_weight",
            self._obj._imaging_weight,  # pylint: disable=protected-access
        )

        # The baselines coord now is missing the antenna1, antenna2 keys
        # so we add those back
        def generate_baselines(baseline_id):
            for a1 in baseline_id:
                for a2 in baseline_id:
                    if a2 >= a1:
                        yield a1, a2

        sub_bvis["baselines"] = pandas.MultiIndex.from_tuples(
            generate_baselines(ids),
            names=("antenna1", "antenna2"),
        )
        return sub_bvis


class FlagTable(xarray.Dataset):
    """Flag table class

    Flags, time, integration_time, frequency, channel_bandwidth, pol,
    in the format of a xarray.

    The configuration is also an attribute.
    """

    __slots__ = ()

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(
        cls,
        baselines=None,
        flags=None,
        frequency=None,
        channel_bandwidth=None,
        configuration=None,
        time=None,
        integration_time=None,
        polarisation_frame=None,
    ):
        """FlagTable

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param configuration: Configuration
        :param time: Time (UTC) [ntimes]
        :param flags: Flags [ntimes, nbaseline, nchan]
        :param integration_time: Integration time [ntimes]
        """
        # pylint: disable=duplicate-code
        coords = {
            "time": time,
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
        }

        datavars = {}
        datavars["flags"] = xarray.DataArray(
            flags, dims=["time", "baselines", "frequency", "polarisation"]
        )
        datavars["integration_time"] = xarray.DataArray(
            integration_time, dims=["time"]
        )
        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"]
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims="time",
        )

        attrs = {}
        attrs["data_model"] = "FlagTable"
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["configuration"] = configuration  # Antenna/station configuration

        return cls(datavars, coords=coords, attrs=attrs)

    def __sizeof__(self):
        """Override default method to return size of dataset
        :return: int
        """
        # Dask uses sizeof() class to get memory occupied by various data
        # objects. For custom data objects like this one, dask falls back to
        # sys.getsizeof() function to get memory usage. sys.getsizeof() in
        # turns calls __sizeof__() magic method to get memory size. Here we
        # override the default method (which gives size of reference table)
        # to return size of Dataset.
        return int(self.nbytes)

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy FlagTable

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set flags to zero in copied object
        """
        new_ft = super().copy(deep=deep, data=data)
        if zero:
            new_ft.data["flags"][...] = 0
        return new_ft


@xarray.register_dataset_accessor("flagtable_acc")
class FlagTableAccessor(XarrayAccessorMixin):
    """
    FlagTable property accessor.
    """

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return self.polarisation_frame.npol

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self._obj.attrs["configuration"].configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self._obj.coords["baselines"])

    def qa_flag_table(self, context=None) -> QualityAssessment:
        """Assess the quality of FlagTable

        :param context:
        :param ft: FlagTable to be assessed
        :return: QualityAssessment
        """
        aflags = numpy.abs(self._obj.flags)
        data = {
            "maxabs": numpy.max(aflags),
            "minabs": numpy.min(aflags),
            "mean": numpy.mean(aflags),
            "sum": numpy.sum(aflags),
            "medianabs": numpy.median(aflags),
        }
        qa = QualityAssessment(
            origin="qa_flagtable", data=data, context=context
        )
        return qa
