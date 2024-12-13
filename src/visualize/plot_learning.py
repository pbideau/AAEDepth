import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab

filename = '/home/Pia/Development/event-camera/x-student-project/x-student-events/pias_loss_list_cosAnneal_prior.txt'
filename_new = '/home/Pia/Development/event-camera/x-student-project/x-student-events/pias_loss_list.txt'

#filename = '/home/Pia/Development/event-camera/x-student-project/x-student-events/cmax_loss_list_orig.txt'
#filename_new = '/home/Pia/Development/event-camera/x-student-project/x-student-events/cmax_loss_list_50_2.txt'

loss_data = np.loadtxt(filename)
loss_data_2 = np.loadtxt(filename_new)

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20,10))
for i, ax in enumerate(axes.flatten()):
    ax.set_yscale('log')
    a1 = np.amax(loss_data[i, :])
    a2 = np.amax(loss_data_2[i, :])
    a = np.maximum(a1,a2)
    b1 = np.amin(loss_data[i, :])
    b2 = np.amin(loss_data_2[i, :])
    b = np.minimum(b1,b2)
    dif = a-b
    ax.set_ylim([b, a])
    #ax.set_ylim([b-dif*0.0001, b+dif*0.01])
    ax.set_xlim([1, 100])
    ax.plot(loss_data[i, :])
    ax.plot(loss_data_2[i, :])

plt.show()