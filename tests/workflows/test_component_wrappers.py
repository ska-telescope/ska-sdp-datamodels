""" Unit tests for json helpers


"""
import os
import unittest

from data_models.parameters import arl_path
from workflows.wrappers.component_wrapper import component_wrapper


class TestComponentWrappers(unittest.TestCase):
    
    def test_run_components(self):
        files = ["test_results/test_pipeline.log",
                 "test_results/test_skymodel.hdf",
                 "test_results/test_empty_vislist.hdf",
                 "test_results/test_perfect_vislist.hdf",
                 "test_results/test_perfect_restored.fits",
                 "test_results/test_perfect_deconvolved.fits",
                 "test_results/test_perfect_residual.fits"
                 ]
        try:
            for f in files:
                os.remove(arl_path(f))
        except FileNotFoundError:
            pass
        
        config_files = ["test_create_vislist.json",
                        "test_create_skymodel.json",
                        "test_predict_vislist.json",
                        "test_continuum_imaging.json"]
        
        for config_file in config_files:
            component_wrapper(config_file)
        
        for f in files:
            assert os.path.isfile(arl_path(f)), "File %s does not exist" % arl_path(f)


if __name__ == '__main__':
    unittest.main()
