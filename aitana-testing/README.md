# aitana-testing

Synthetic seismic waveforms with known spectral content, shared by the tests of the
aitana packages and by `aitana.access.seismic.MockSDSWaveforms`.

```python
from aitana.testing import synthetic_trace

tr = synthetic_trace(nsec=600, gaps=True)  # obspy.Trace
```
