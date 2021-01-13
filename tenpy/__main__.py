"""The tenpy entry point, called by ``python -m tenpy``."""
# Copyright 2020-2021 TeNPy Developers, GNU GPLv3

import tenpy

if __name__ == "__main__":
    import sys
    raise SystemExit(tenpy.console_main())
