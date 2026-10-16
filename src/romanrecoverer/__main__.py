"""Allow ``python -m romanrecoverer``."""

from __future__ import annotations

import sys

from romanrecoverer.cli import main

if __name__ == "__main__":
    sys.exit(main())
