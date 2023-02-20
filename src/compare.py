# Standard library imports
import argparse
import os
from typing import Tuple, List, Union, Callable

# Third-party related imports
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

#
from plotting import *


# ARGUMENT PARSING
parser = argparse.ArgumentParser(
        prog = "CompareResults",
        description = "Run comparisons between NeRF and DS-NeRF")

parser.add_argument("-d1", "--dir1", dest="dir1", required=True, type=str,
                    help="Base path for NeRF results")
parser.add_argument("-d2", "--dir2", dest="dir2", required=True, type=str,
                    help="Base path for DS=NeRF results")

args = parser.parse_args()

# load .npy files
curves1 = np.load(args.dir1)
curves2 = np.load(args.dir2)

# call animation fn and save returned animation to .gif format
anim = density_animate(curves1, curves2)
anim.save(filename="./densities.gif", writer="pillow")
