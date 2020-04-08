import sys
from absl import flags
import re
import numpy as np
import matplotlib.pyplot as plt
flags.DEFINE_string('input_file', '', 'file for input data')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
new_loss = []
origin_loss = []
steps = []
with open(FLAGS.input_file) as f:
    lines = f.readlines()
    for line in lines:
        num = re.findall(r"\d+\.?\d*", line)
        if len(num) == 0:
            break
        steps.append(int(num[0]))
        origin_loss.append(float(num[1]))
        new_loss.append(float(num[2]))

total_width, n = 0.8, 2
width = total_width / n
plt.figure()

x = list(range(len(new_loss)))
plt.bar(x, origin_loss, width=width, label="origin")
for i in range(len(x)):
    x[i] = x[i] + width/2
plt.bar(x, new_loss, width=width, label="new", tick_label = steps)
plt.xlabel("iterations", fontsize=18)
plt.ylabel('test loss', fontsize=18)
plt.legend(fontsize=18)
plt.show()
