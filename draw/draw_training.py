import sys
from absl import flags
import re
import numpy as np
import matplotlib.pyplot as plt
flags.DEFINE_string('input_file', '', 'file for input data')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
loss = []
steps = []
with open(FLAGS.input_file) as f:
    lines = f.readlines()
    for line in lines:
        num = re.findall(r"\d+\.?\d*", line)
        if len(num) == 0:
            break
        num = num[0:2]
        loss.append(float(num[0]))
        steps.append(float(num[1]))
plt.figure()
all_iteration = int(steps[-1])
interval = int(all_iteration/1000 + 1)
new_ticks = np.linspace(0, all_iteration, interval)
plt.xticks(new_ticks)
new_ticks_ = np.linspace(0, 2.5, 6)
plt.yticks(new_ticks_)
plt.plot(steps, loss)
plt.xlabel("iterations")
plt.ylabel('training loss')

start = [0, loss[0]]
end = [all_iteration, loss[-1]]

plt.plot([start[0], end[0]], [start[1], end[1]], c='r', linestyle="--")
plt.scatter([start[0], end[0]], [start[1], end[1]], c='r', s=30, alpha=1)
plt.annotate(s=str(start[1]), xy=(start[0], start[1]),
             xytext=(-20, 10), textcoords='offset points')
plt.annotate(s=str(end[1]), xy=(end[0], end[1]),
             xytext=(-20, 10), textcoords='offset points')
plt.show()

