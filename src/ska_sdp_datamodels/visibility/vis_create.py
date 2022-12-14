"""
Functions to create Visibility
"""

import logging

import numpy
import pandas
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
    xyz_at_latitude,
    xyz_to_uvw,
)
from ska_sdp_datamodels.physical_constants import SIDEREAL_DAY_SECONDS
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    correlate_polarisation,
)
from ska_sdp_datamodels.visibility.vis_model import FlagTable, Visibility
from ska_sdp_datamodels.visibility.vis_utils import (
    calculate_transit_time,
    generate_baselines,
)

log = logging.getLogger("data-models-logger")


# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-branches,too-many-statements
def create_visibility(
    config: Configuration,
    times: numpy.array,
    frequency: numpy.array,
    phasecentre: SkyCoord,
    channel_bandwidth: numpy.array,
    weight: float = 1.0,
    polarisation_frame: PolarisationFrame = None,
    integration_time=1.0,
    zerow=False,
    elevation_limit=15.0 * numpy.pi / 180.0,
    source="unknown",
    meta=None,
    utc_time=None,
    times_are_ha=True,
) -> Visibility:
    """
    Create a Visibility from Configuration, hour angles,
    and direction of source

    Note that we keep track of the integration time for BDA purposes

    The input times are hour angles in radians, these are
    converted to UTC MJD in seconds, using utc_time as
    the approximate time.

    :param config: Configuration of antennas
    :param times: time or hour angles in radians
    :param frequency: frequencies (Hz] [nchan]
    :param phasecentre: phasecentre of observation (SkyCoord)
    :param channel_bandwidth: channel bandwidths: (Hz] [nchan]
    :param weight: weight of a single sample
    :param polarisation_frame: PolarisationFrame('stokesI')
    :param integration_time: Integration time ('auto' or value in s)
    :param zerow: bool - set w to zero
    :param elevation_limit: in degrees
    :param source: Source name
    :param meta: Meta data as a dictionary
    :param utc_time: Time of ha definition default is
                Time("2000-01-01T00:00:00", format='isot', scale='utc')
    :param times_are_ha: The times are hour angles (default)
                instead of utc time (in radians)
    :return: Visibility
    """
    if phasecentre is None:
        raise ValueError("Must specify phase centre")

    if utc_time is None:
        utc_time_zero = Time("2000-01-01T00:00:00", format="isot", scale="utc")
    elif isinstance(utc_time, Time):
        utc_time_zero = utc_time
        utc_time = None

    if polarisation_frame is None:
        polarisation_frame = correlate_polarisation(config.receptor_frame)

    latitude = config.location.geodetic[1].to("rad").value
    ants_xyz = config["xyz"].data
    ants_xyz = xyz_at_latitude(ants_xyz, latitude)
    nants = len(config["names"].data)

    baselines = pandas.MultiIndex.from_tuples(
        generate_baselines(nants), names=("antenna1", "antenna2")
    )
    nbaselines = len(baselines)

    # Find the number of integrations ab
    ntimes = 0
    n_flagged = 0

    for itime, time in enumerate(times):

        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        if times_are_ha:
            ha = time
        else:
            ha = time * (SIDEREAL_DAY_SECONDS / 86400.0)

        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            # check if at this time target is above an elevation_limit
            ntimes += 1
        else:
            # if target is below, flag it
            n_flagged += 1

    if ntimes == 0:
        raise ValueError(
            "No targets above elevation_limit; all points are flagged"
        )

    if elevation_limit is not None and n_flagged > 0:
        log.info(
            "create_visibility: flagged %d/%d times "
            "below elevation limit %f (rad)",
            n_flagged,
            ntimes,
            180.0 * elevation_limit / numpy.pi,
        )
    else:
        log.debug("create_visibility: created %d times", ntimes)

    npol = polarisation_frame.npol
    nchan = len(frequency)
    visshape = [ntimes, nbaselines, nchan, npol]
    rvis = numpy.zeros(visshape, dtype="complex")
    rflags = numpy.ones(visshape, dtype="int")
    rweight = numpy.ones(visshape)
    rtimes = numpy.zeros([ntimes])
    rintegrationtime = numpy.zeros([ntimes])
    ruvw = numpy.zeros([ntimes, nbaselines, 3])

    if utc_time is None:
        stime = calculate_transit_time(
            config.location, utc_time_zero, phasecentre
        )
        if stime.masked:
            stime = utc_time_zero

    # Do each time filling in the actual values
    itime = 0
    for _, time in enumerate(times):

        if times_are_ha:
            ha = time
        else:
            ha = time * (SIDEREAL_DAY_SECONDS / 86400.0)

        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            rtimes[itime] = stime.mjd * 86400.0 + time * 86400.0 / (
                2.0 * numpy.pi
            )
            rweight[itime, ...] = 1.0
            rflags[itime, ...] = 1

            # Loop over all pairs of antennas. Note that a2>a1
            ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)

            for ibaseline, (a1, a2) in enumerate(baselines):
                if a1 != a2:
                    rweight[itime, ibaseline, ...] = weight
                    rflags[itime, ibaseline, ...] = 0
                else:
                    rweight[itime, ibaseline, ...] = 0.0
                    rflags[itime, ibaseline, ...] = 1

                ruvw[itime, ibaseline, :] = ant_pos[a2, :] - ant_pos[a1, :]
                rflags[itime, ibaseline, ...] = 0

            if itime > 0:
                rintegrationtime[itime] = rtimes[itime] - rtimes[itime - 1]
            itime += 1

    if itime > 1:
        rintegrationtime[0] = rintegrationtime[1]
    else:
        rintegrationtime[0] = integration_time
    rchannel_bandwidth = channel_bandwidth
    if zerow:
        ruvw[..., 2] = 0.0

    vis = Visibility.constructor(
        uvw=ruvw,
        time=rtimes,
        frequency=frequency,
        vis=rvis,
        weight=rweight,
        baselines=baselines,
        flags=rflags,
        integration_time=rintegrationtime,
        channel_bandwidth=rchannel_bandwidth,
        polarisation_frame=polarisation_frame,
        source=source,
        meta=meta,
        phasecentre=phasecentre,
        configuration=config,
    )

    log.debug(
        "create_visibility: %d rows, %.3f GB",
        vis.visibility_acc.nvis,
        vis.visibility_acc.size(),
    )

    return vis


def create_flagtable_from_visibility(vis: Visibility) -> FlagTable:
    """
    Create FlagTable matching Visibility

    :param vis: Visibility object
    :return: FlagTable object
    """
    return FlagTable.constructor(
        flags=vis.flags,
        frequency=vis.frequency,
        channel_bandwidth=vis.channel_bandwidth,
        configuration=vis.configuration,
        time=vis.time,
        integration_time=vis.integration_time,
        polarisation_frame=vis.visibility_acc.polarisation_frame,
    )
