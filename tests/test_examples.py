"""check whether the examples run without problems"""

import sys
import os

# get directory where the examples can be found
ex_dir = os.path.join(os.path.dirname(__file__), '../examples')
sys.path[:0] = [ex_dir]  # add it to sys.path


def test_npc_intro():
    print sys.path
    import npc_intro  # the examples are not protected by ``if __name__ == "__main__"``
