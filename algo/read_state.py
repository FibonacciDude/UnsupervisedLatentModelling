import numpy as np
import torch
import gym
import matplotlib.pyplota as plt
from env import *

plots = 

curr_pos = 0
def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)

    ax.cla()
    ax.set_title(plots[curr_pos][1])
    #ax.imshow(plots[curr_pos][0])
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.plot(t,y1)
plt.show()
