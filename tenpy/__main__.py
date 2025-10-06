"""The tenpy entry point, called by ``python -m tenpy``."""
# Copyright (C) TeNPy Developers, Apache license

import tenpy

if __name__ == "__main__":
    import sys  # noqa 401
    tenpy.console_main()
    sys.exit(0)
