#!/usr/bin/python

from sys import *
from pylab import *
from matplotlib import patches
import numpy as np

state_array = []
NUM_DIM = 2
NUM_STATES = 0

if len(argv) > 1:
    save_name = argv[1]
else:
    save_name = "none"

def wstd(x,w):
    t = w.sum()
    return (((w*x**2).sum()*t-(w*x).sum()**2)/(t**2-(w**2).sum()))**.5

def read_state_index():
    global state_array, NUM_DIM, NUM_STATES
    
    states = open("state_index.dat", 'r')
    
    if states:
        lines = states.readlines()
        for l in lines:
            s = l.split('\t')
            to_put = [float(s[x]) for x in range(NUM_DIM+1)]
            state_array.append(to_put)
    
    states.close()

    state_array = np.array(state_array)
    state_array = state_array[:,1:]     # remove index
    NUM_STATES = len(state_array[:,0])
    print "num_states: ", NUM_STATES

def read_traj():

    fp = open("singleint_sim.dat", 'r')
    if fp:
        tmp = fp.readline()
        lines = fp.readlines()

        belief_traj = []
        to_put = 0.0*np.arange(NUM_STATES)

        for l in lines:
            s = l.split('\t')
            if len(s) == 1:
                belief_traj.append(to_put)
                to_put = 0.0*np.arange(NUM_STATES)
            elif(len(s) == 2):
                a =  int(s[0])
                #print a, float(s[1])
                to_put[a] = float(s[1])
    
    fp.close()

    #print belief_traj
    belief_traj = np.array(belief_traj)
    
    traj_xy = []
    traj_std = []
    num_traj = len(belief_traj[:,0])
    for i in range(num_traj):
        
        mx = np.average(state_array[:,0], weights=belief_traj[i,:])
        my = np.average(state_array[:,1], weights=belief_traj[i,:])
        traj_xy.append([mx, my])
        
        stdx = wstd(state_array[:,0], belief_traj[i,:])
        stdy = wstd(state_array[:,1], belief_traj[i,:])
        traj_std.append([stdx, stdy])

    fig = figure(1)
    traj_xy = np.array(traj_xy)
    traj_std = np.array(traj_std)
    ax = fig.add_subplot(111)
    """
    for i in range(num_traj):
        ellipse = patches.Ellipse( (traj_xy[i,0], traj_xy[i,1]), 2*traj_std[i,0], 2*traj_std[i,1], fc='blue', alpha = 0.2)
        ax.add_patch(ellipse)
    """
    plot(traj_xy[:,0], traj_xy[:,1], 'bo-')
    grid()
    show()
    fig.savefig(save_name)

if __name__ == "__main__":
    
    read_state_index()
    read_traj()
