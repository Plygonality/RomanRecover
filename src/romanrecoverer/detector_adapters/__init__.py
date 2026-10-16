"""External detector and instrument adapters.

Nothing in this package reimplements a source detector or an isolation forest.
"""

from romanrecoverer.detector_adapters.ngrst import probe_ngrst
from romanrecoverer.detector_adapters.romanisim_l2 import probe_romanisim

__all__ = ["probe_ngrst", "probe_romanisim"]
