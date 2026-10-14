import unittest

import numpy as np

from aitana.features.spectral_features import SpectralFeatures
from aitana.testing import synthetic_trace


class SpectralTestCase(unittest.TestCase):
    sup = np.testing.suppress_warnings()
    sup.filter(module=np.matrixlib.defmatrix)  # module must match exactly

    def test_spectral(self):
        sf = SpectralFeatures(filtertype="bandpass", filterfreq=(0.5, 25))
        test_trace = synthetic_trace(
            addchirp=False,
            frequencies=[5.0],
            amplitudes=[1.0],
            phases=[0.0],
            sinusoid=False,
            noise=False,
        )
        xds = sf.compute(test_trace)
        self.assertAlmostEqual(xds.predom_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.central_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.bandwidth.data[0], 0.062, 3)

    def test_spectral_with_gaps(self):
        sf = SpectralFeatures(filtertype="bandpass", filterfreq=(0.5, 25))
        test_trace = synthetic_trace(
            addchirp=False,
            frequencies=[5.0],
            amplitudes=[1.0],
            phases=[0.0],
            sinusoid=False,
            noise=False,
            gaps=True,
        )
        xds = sf.compute(test_trace)
        self.assertAlmostEqual(xds.predom_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.central_freq.data[0], 5, 0)
        self.assertAlmostEqual(xds.bandwidth.data[0], 0.062, 3)
        self.assertEqual(np.isnan(xds.predom_freq.values).sum(), 3)
