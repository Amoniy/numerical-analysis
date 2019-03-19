import matplotlib.pyplot as plt
import numpy as np
import math
import os

from first_task import transformer
from first_task import rootpath

fileName = "4k"
inputFileName = os.path.join(rootpath.rootpath, "resources/" + fileName + ".txt")
outputFileName = os.path.join(rootpath.rootpath, "output/" + fileName + ".png")

data = np.array(transformer.transformInputFile(inputFileName))

pixels = 8192 // 2

imgSize = math.ceil(pixels / 100)
plt.figure(figsize=(imgSize, imgSize))
plt.xlabel('real axis')
plt.ylabel('imaginary axis')

tickCount = 16
rounding = 4
shift = 4 / tickCount
blockSize = 4 / pixels

tickNames = []
for index in range(0, tickCount + 1):
    tickNames.append(round((-2 + index * shift), rounding))

tickIndexes = []
for index in range(0, tickCount + 1):
    tickIndexes.append(index * pixels / tickCount)

plt.xticks(tickIndexes, tickNames)
plt.yticks(tickIndexes, reversed(tickNames))

plt.imsave(outputFileName, data)

plt.show()
