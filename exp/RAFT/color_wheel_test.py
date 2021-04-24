import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

from core.utils import flow_viz

clr_wheel= np.uint8(flow_viz.make_colorwheel().reshape(1, 55, 3))
clr_wheel= Image.fromarray(clr_wheel)
clr_wheel= clr_wheel.resize((5500, 100), resample= Image.BOX)

clr_wheel.save('test.png')