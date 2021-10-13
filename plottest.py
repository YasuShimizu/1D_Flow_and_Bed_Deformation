from matplotlib import pyplot as plt
from random import randint
import numpy as np

# 
x = list(range(10))
y = list(range(10))

for i in np.arange(0,10):
    y[i]=np.random.rand()


# グラフの描画
plt.plot(x, y)
plt.show()
