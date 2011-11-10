#!/usr/bin/python

import time
from sys import *
from pylab import *
from matplotlib import patches
import matplotlib.colors as mcolor
import matplotlib.cm as cm
import numpy as np

state_array = []
NUM_DIM = 2
NUM_STATES = 0

nf1, nf2 = "none", "none"
if len(argv) == 1:
    nf1 = "none"
    nf2 = "none"
elif len(argv) <= 2:
    nf1 = argv[1]
elif len(argv) <= 3:
    nf2 = argv[2]

def wstd(x,w):
    t = w.sum()
    return (((w*x**2).sum()*t-(w*x).sum()**2)/(t**2-(w**2).sum()))**.5

def read_state_trajectories():
    global state_array, NUM_DIM, NUM_STATES
    
    state_trajs = []
    to_put = []

    trajs = open("state_trajectories.dat", 'r')
    if trajs:
        lines = trajs.readlines()
        for l in lines:
            s = l.split('\t')
            
            num_steps = len(s) -1
            if( len(s) > 3):
                to_put = [int(s[x]) for x in range(num_steps)]
                state_trajs.append(to_put)

    trajs.close()

    state_trajs = np.array(state_trajs)
    num_traj = len(state_trajs)
    traj_len = len(state_trajs[0])
    print "num_traj: ", num_traj, " traj_len: ", traj_len

    """
    fig = figure(3)
    fig.add_subplot(111, aspect='equal')
    # create a hexbin map now for each trajectory
    for i in range(len(state_trajs)):
        curr_traj = np.array([ [state_array[x,0], state_array[x,1]] for x in state_trajs[i] ] )
        clf()
        scatter( curr_traj[:,0], curr_traj[:,1], marker='o', c='y', s= 25, alpha=0.7)
        #hexbin(curr_traj[:,0], curr_traj[:,1], gridsize=10, cmap=cm.get_cmap('Jet'), alpha=0.9, mincnt=1)
        fig.savefig("movie/"+str(i)+".png")
    """
    
    fig = figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(len(state_trajs)):
        curr_traj = np.array([ [state_array[x,0], state_array[x,1]] for x in state_trajs[i] ] )
        plot(curr_traj[:,0], curr_traj[:,1], 'b-', lw=0.5, alpha=0.2)
        
        circle = Circle( (curr_traj[0,0], curr_traj[0,1]), 0.01, fc='red', alpha = 0.4)
        ax.add_patch(circle)
        circle = Circle( (curr_traj[traj_len-1,0], curr_traj[traj_len-1,1]), 0.01, fc='green', alpha = 0.4)
        ax.add_patch(circle)
    
    fig = figure(2)
    state_traj_x = []
    state_traj_y = []

    for i in range(num_traj):
        tmp_traj = []
        for x in state_trajs[i]:
            #if x not in tmp_traj:
            tmp_traj.append(x)

        curr_traj = np.array([ [state_array[x,0], state_array[x,1]] for x in tmp_traj ] )
        tmp = np.array( [state_array[x,0] for x in tmp_traj])
        state_traj_x.append(tmp)
        tmp = np.array( [state_array[x,1] for x in tmp_traj])
        state_traj_y.append(tmp)

        #subplot(211)
        #plot(curr_traj[:,0], 'b-', lw=0.5, alpha=0.10)
        #subplot(212)
        #plot(curr_traj[:,1], 'b-', lw=0.5, alpha=0.10)
    
    state_traj_x = np.array(state_traj_x)
    state_traj_y = np.array(state_traj_y)
    
    subplot(211)
    errorbar( np.linspace(0,traj_len,num=traj_len), np.average(state_traj_x, axis=0), yerr=np.std(state_traj_x, axis=0), fmt='r-', ecolor='red')
    subplot(212)
    errorbar( np.linspace(0,traj_len,num=traj_len), np.average(state_traj_y, axis=0), yerr=np.std(state_traj_y, axis=0), fmt='r-', ecolor='red')

    subplot(211)
    grid()
    subplot(212)
    grid()

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

    fig = figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    scatter( state_array[:,0], state_array[:,1], c='y', marker='o', s=30, alpha=0.8)

def draw_goal():
    
    fig = figure(1)
    ax = fig.add_subplot(111, aspect='equal')

    rect = Rectangle( (0.3, 0), 0.2, 0.2, fc='green', alpha = 0.4)
    ax.add_patch(rect)

def read_belief_traj():

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
    ax = fig.add_subplot(111, aspect='equal')

    circle = Circle( (traj_xy[0,0], traj_xy[0,1]), 0.01, fc='red', alpha = 0.4)
    ax.add_patch(circle)
    circle = Circle( (traj_xy[num_traj-1,0], traj_xy[num_traj-1,1]), 0.01, fc='green', alpha = 0.4)
    ax.add_patch(circle)

    plot(traj_xy[:,0], traj_xy[:,1], 'b-')
    for i in range(num_traj):
        circle = Circle( (traj_xy[i,0], traj_xy[i,1]), traj_std[i,0]/4, fc='blue', alpha = 0.05)
        ax.add_patch(circle)
        

if __name__ == "__main__":
    
    read_state_index()
    read_state_trajectories()
    #draw_goal()
    
    fig = figure(1)
    grid()
    if(nf1 != "none"):
        fig.savefig(nf1)
    
    fig = figure(2)
    if(nf2 != "none"):
        fig.savefig(nf2)

    if( (nf1 == "none") or (nf2 =="none")):
        show()
