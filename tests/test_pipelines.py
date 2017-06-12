"""Unit tests for pipelines


"""

import unittest

from arl.fourier_transforms.ftprocessor import *
from arl.image.operations import export_image_to_fits
from arl.pipelines.functions import *
from arl.skycomponent.operations import create_skycomponent
from arl.util.testing_support import create_named_configuration, create_test_image, create_blockvisibility_iterator


class TestPipelines(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        lowcore = create_named_configuration('LOWBD2-CORE')
        times = numpy.arange(-3.0, +3.0, 1.0) * numpy.pi / 12.0
        vnchan = 3
        frequency = numpy.linspace(0.8e8, 1.20e8, vnchan)
        channel_bandwidth = numpy.array(vnchan * [frequency[1] - frequency[0]])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox=2000.0)
        self.compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox=2000.0)
        
        self.comp = create_skycomponent(flux=self.flux, frequency=frequency, direction=self.compabsdirection)
        self.image = create_test_image(frequency=frequency, phasecentre=self.phasecentre, cellsize=0.001,
                                       polarisation_frame=PolarisationFrame('stokesIQUV'))
        
        self.blockvis = create_blockvisibility_iterator(lowcore, times=times, frequency=frequency,
                                                        channel_bandwidth=channel_bandwidth,
                                                        phasecentre=self.phasecentre, weight=1,
                                                        polarisation_frame=PolarisationFrame('linear'),
                                                        integration_time=1.0, number_integrations=1, predict=predict_2d,
                                                        components=self.comp, phase_error=0.1, amplitude_error=0.01,
                                                        sleep=1.0)
        
        self.vis = create_blockvisibility(lowcore, times=times, frequency=frequency,
                                     channel_bandwidth=channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1,
                                     polarisation_frame=PolarisationFrame('stokesIQUV'),
                                     integration_time=1.0)
        
        self.vis = predict_2d(self.vis, self.image)
    
    def ingest(self):
        vis = None
        for iv, subvis in enumerate(self.blockvis):
            if iv == 0:
                vis = subvis
            else:
                vis = append_visibility(vis, subvis)
        
        return vis
    
    def test_RCAL(self):
        for igt, gt in enumerate(rcal(vis=self.blockvis, components=self.comp)):
            log.info("Chunk %d, gaintable size %.3f (GB)" % (igt, gt.size()))
    
    def test_ICAL(self):
        icalpipe = ical(vis=self.vis, components=self.comp)
    
    def test_continuum_imaging(self):
        model = create_empty_image_like(self.image)
        visres, comp, residual = continuum_imaging(self.vis, model, algorithm='msmfsclean')
        export_image_to_fits(comp, "%s/test_pipelines-continuum-imaging-comp.fits" % (self.dir))
    
    def test_spectral_line_imaging_no_deconvolution(self):
        model = create_empty_image_like(self.image)
        visres, comp, residual = spectral_line_imaging(self.vis, model, continuum_model=model,
                                                       deconvolve_spectral=False)
        export_image_to_fits(comp, "%s/test_pipelines-spectral-no-deconvolution-imaging-comp.fits" % (self.dir))

    def test_spectral_line_imaging_with_deconvolution(self):
        model = create_empty_image_like(self.image)
        visres, comp, residual = spectral_line_imaging(self.vis, model, continuum_model=self.image, algorithm='hogbom',
                                                       deconvolve_spectral=True)
        export_image_to_fits(comp, "%s/test_pipelines-spectral-with-deconvolution-imaging-comp.fits" % (self.dir))

    def test_fast_imaging(self):
        fipipe = fast_imaging(vis=self.vis, components=self.comp, Gsolinit=300.0)
    
    def test_EOR(self):
        eorpipe = eor(vis=self.vis, components=self.comp, Gsolinit=300.0)


if __name__ == '__main__':
    unittest.main()
