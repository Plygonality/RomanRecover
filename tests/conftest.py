"""Keep Astropy from touching the network during WCS tests."""

from astropy.utils import iers

iers.conf.auto_download = False
