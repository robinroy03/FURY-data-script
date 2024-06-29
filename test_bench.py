from test_code import scene
from fury import window
import numpy.testing as npt
import numpy as np

window.record(scene, out_path="output.png")