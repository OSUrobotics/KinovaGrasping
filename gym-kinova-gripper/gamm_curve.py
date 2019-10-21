#!/usr/biin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
a = 4
x = np.arange(0.0, 0.71, step=0.01)
gamma = 0.01
# x = x[::-1]
# y = a * (1-x) ** (gamma)
# y = np.sqrt(1 - x**3) 
y = np.tanh(1 - x)


plt.plot(x, y)
plt.show()