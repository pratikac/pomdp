#!/usr/bin/python

import time
from sys import *
from pylab import *
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolor
import matplotlib.cm as cm
import numpy as np
import matplotlib.mlab as mlab
import os

holding_time = 1
state_array = []
NUM_DIM = 2
NUM_STATES = -1
TRAJ_LEN = 10

nf1, nf2 = "none", "none"
if len(argv) == 1:
    nf1 = "none"
    nf2 = "none"
elif len(argv) <= 2:
    nf1 = argv[1]
elif len(argv) <= 3:
    nf1 = argv[1]
    nf2 = argv[2]

def wstd(x,w):
    t = w.sum()
    return (((w*x**2).sum()*t-(w*x).sum()**2)/(t**2-(w**2).sum()))**.5

def read_lqg_trajectories():
    lqg_trajs = []
    to_put = []
    costs = []

    trajs = open("lqg_trajectories.dat", 'r')
    if trajs:
        lines = trajs.readlines()
        for l in lines:
            s = l.split('\t')
            if (s[0] == 's:'):
                if len(to_put) != 0:
                    lqg_trajs.append(to_put)
                    to_put = []
                    costs.append(float(s[2]))
            else:
                to_put.append([float(s[x]) for x in range(NUM_DIM)])

    trajs.close()
    
    if len(lqg_trajs):
        lqg_trajs = np.array(lqg_trajs)
        num_lqg = len(lqg_trajs)
        lqg_len = len(lqg_trajs[0])
        print "num_lqg: ", num_lqg, " lqg_len: ", lqg_len, "cost: ", np.average(costs)

        if NUM_DIM == 2:
            lqg_traj_x = np.array(lqg_trajs[:,:,0])
            lqg_traj_y = np.array(lqg_trajs[:,:,1])

            figure(2)
            subplot(211)
            errorbar( np.linspace(0,lqg_len,num=lqg_len), np.average(lqg_traj_x, axis=0), yerr=np.std(lqg_traj_x, axis=0), fmt='r-', ecolor='red')

            #subplot(212)
            #errorbar( np.linspace(0,lqg_len,num=lqg_len), np.average(lqg_traj_y, axis=0), yerr=np.std(lqg_traj_y, axis=0), fmt='r-', ecolor='red')
        elif NUM_DIM==1:
            lqg_traj_x = np.array(lqg_trajs[:,:,0])

            figure(2)
            subplot(111)
            tmp1 = np.average(lqg_traj_x, axis=0) + np.std(lqg_traj_x, axis=0)
            tmp2 = np.average(lqg_traj_x, axis=0) - np.std(lqg_traj_x, axis=0)
            plot(np.linspace(0,lqg_len,num=lqg_len), np.average(lqg_traj_x, axis=0), 'r-', label="LQG_mean")
            plot(np.linspace(0,lqg_len,num=lqg_len), tmp1, 'r--', label="LQG variance")
            plot(np.linspace(0,lqg_len,num=lqg_len), tmp2, 'r--')
            legend()

def read_state_trajectories():
    global state_array, NUM_DIM, NUM_STATES, TRAJ_LEN
    
    state_trajs = []
    to_put = []
    rewards = []

    trajs = open("state_trajectories.dat", 'r')
    if trajs:
        lines = trajs.readlines()
        for l in lines:
            s = l.split('\t')
            
            num_steps = len(s) -1
            if( len(s) > 3):
                to_put = [int(s[x]) for x in range(num_steps)]
                state_trajs.append(to_put)
                
                if len(to_put) > TRAJ_LEN:
                    TRAJ_LEN = len(to_put)
            else:
                rewards.append(float(s[1]))

    trajs.close()
    
    num_traj = len(state_trajs)
    for ct in range(num_traj):
        last = state_trajs[ct][-1]
        curr_len = len(state_trajs[ct])
        for ti in np.linspace(curr_len, TRAJ_LEN-1, TRAJ_LEN-curr_len):
            state_trajs[ct].append(last)

    state_trajs = np.array(state_trajs)
    traj_len = len(state_trajs[0])
    print "num_traj: ", num_traj , " traj_len: ", traj_len, " reward: ", np.average(rewards)
    
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
    
    """
    if NUM_DIM==2:
        fig = figure(1)
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(len(state_trajs)):
            curr_traj = np.array([ [state_array[x,i] for i in range(NUM_DIM)] for x in state_trajs[i] ] )

            plot(curr_traj[:,0], curr_traj[:,1], 'b-', lw=0.5, alpha=0.2)

            circle = Circle( (curr_traj[0,0], curr_traj[0,1]), 0.01, fc='red', alpha = 0.4)
            ax.add_patch(circle)
            circle = Circle( (curr_traj[traj_len-1,0], curr_traj[traj_len-1,1]), 0.01, fc='green', alpha = 0.4)
            ax.add_patch(circle)
    """

    fig = figure(2)
    state_traj_x = []
    state_traj_y = []

    for i in range(num_traj):
        curr_traj = np.array([ [state_array[x,j] for j in range(NUM_DIM)] for x in state_trajs[i] ] )
        
        tmp = np.array([state_array[x,0] for x in state_trajs[i]])
        state_traj_x.append(tmp)
        
        if NUM_DIM == 2:
            tmp = np.array([state_array[x,1] for x in state_trajs[i]])
            state_traj_y.append(tmp)

        #subplot(111)
        #plot(curr_traj[:,0], 'b-', lw=0.5, alpha=0.10)
        #subplot(212)
        #plot(curr_traj[:,1], 'ro', lw=0.5, alpha=0.05)
    
    state_traj_x = np.array(state_traj_x)
    state_traj_y = np.array(state_traj_y)
    
    print state_traj_x.shape, state_traj_y.shape

    state_traj_x_percentile_10 = np.array([mlab.prctile(state_traj_x[:,i],p=10) for i in range(TRAJ_LEN)])
    state_traj_x_percentile_50 = np.array([mlab.prctile(state_traj_x[:,i],p=50) for i in range(TRAJ_LEN)])
    state_traj_x_percentile_90 = np.array([mlab.prctile(state_traj_x[:,i],p=90) for i in range(TRAJ_LEN)])
    state_traj_x_percentile = np.array([state_traj_x_percentile_10, state_traj_x_percentile_90])

    if NUM_DIM == 2:
        state_traj_y_percentile_10 = np.array([mlab.prctile(state_traj_y[:,i],p=10) for i in range(TRAJ_LEN)])
        state_traj_y_percentile_90 = np.array([mlab.prctile(state_traj_y[:,i],p=90) for i in range(TRAJ_LEN)])
        state_traj_y_percentile = np.array([state_traj_y_percentile_10, state_traj_y_percentile_90])
        
        subplot(211)
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), np.average(state_traj_x, axis=0), 'b-', label='mean')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_x_percentile_10, 'b--', label='10/90 percentile')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_x_percentile_90, 'b--')
        legend()
        grid()
        ylabel('x (t)')
        xlabel('t [s]')
        axis('tight')
    
        subplot(212)
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), np.average(state_traj_y, axis=0), 'b-', label='mean')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_y_percentile_10, 'b--', label='10/90 percentile')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_y_percentile_90, 'b--')
        legend()
        grid()
        ylabel('y (t)')
        xlabel('t [s]')
        axis('tight')

    elif NUM_DIM==1:
        subplot(111)
        #plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), np.average(state_traj_x, axis=0), 'b-', label='mean')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_x_percentile_10, 'b--', label='10/50/90 percentile')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_x_percentile_90, 'b--')
        plot( holding_time*np.linspace(0,TRAJ_LEN,num=TRAJ_LEN), state_traj_x_percentile_50, 'b--')
        legend()
        grid()
        xlabel('t [s]')
        axis('tight')

def read_state_index():
    global state_array, NUM_DIM, NUM_STATES, holding_time
    
    states = open("state_index.dat", 'r')
    
    if states:
        l = states.readline()
        s = l.split('\t')
        holding_time = float(s[0])
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
    print "holding_time: ", holding_time

    fig = figure(3)
    ax = fig.add_subplot(111, aspect='equal')
    scatter( state_array[:,0], state_array[:,1], c='y', marker='o', s=30, alpha=0.8)

def draw_goal():
    
    fig = figure(2)

    if NUM_DIM == 1:
        ax = subplot(111)
        rect = Rectangle( (0, 0.8), holding_time*TRAJ_LEN, 0.2, fc='green', alpha = 0.4)
        ax.add_patch(rect)
        rect = Rectangle( (0, -1), holding_time*TRAJ_LEN, 0.2, fc='red', alpha = 0.4)
        ax.add_patch(rect)
        ylim([-1,1])

    if NUM_DIM == 2:
        ax = subplot(211)
        rect = Rectangle( (0, 0.8), holding_time*TRAJ_LEN, 0.2, fc='green', alpha = 0.4)
        ax.add_patch(rect)
        rect = Rectangle( (0, -1), holding_time*TRAJ_LEN, 0.2, fc='red', alpha = 0.4)
        ax.add_patch(rect)

        ax = subplot(212)
        rect = Rectangle( (0, 0.8), holding_time*TRAJ_LEN, 0.2, fc='green', alpha = 0.4)
        ax.add_patch(rect)
        rect = Rectangle( (0, -1), holding_time*TRAJ_LEN, 0.2, fc='red', alpha = 0.4)
        ax.add_patch(rect)


def read_belief_traj():

    fp = open("belief_trajectory.dat", 'r')
    if fp and os.path.getsize('belief_trajectory.dat') > 0:
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

        belief_traj.append(to_put)

        #print belief_traj
        belief_traj = np.array(belief_traj)

        traj_xy = []
        traj_std = []
        num_traj = len(belief_traj[:,0])
        for i in range(num_traj):
            #print belief_traj[i,:] 
            to_put = [np.average(state_array[:,x], weights=belief_traj[i,:]) for x in range(NUM_DIM)]
            traj_xy.append(to_put)

            stdx = [wstd(state_array[:,x], belief_traj[i,:]) for x in range(NUM_DIM)]
            traj_std.append(stdx)

        traj_xy = np.array(traj_xy)
        traj_std = np.array(traj_std)

        print traj_std.shape, traj_xy.shape

        os.system('rm -r movie/fig* movie/animation.avi')
        if NUM_DIM==1:

            xarr = np.linspace(-1,1,1000)
            for x in range(len(traj_xy)):
                fig = figure(1)
                ax = fig.add_subplot(111, aspect='equal')
                """
                yarr = 0*xarr;
                for i in range(len(xarr)):
                    yarr[i] = 1/sqrt(2*3.1415)/traj_std[x]*exp(-0.5/(traj_std[x]**2)*(xarr[i] - traj_xy[x])**2)
                plot(xarr, yarr, 'b-')
                fill_between(xarr, yarr, 0*yarr, facecolor='blue', alpha=0.3)
                """
                for i in range(NUM_STATES):
                    rect = Rectangle( (state_array[i,0]-0.05,0), 0.1, belief_traj[x,i], fc='blue', alpha=0.3)
                    ax.add_patch(rect)
                
                #rect = Rectangle( (traj_xy[x][0]-0.05, 0), 0.1,1/sqrt(2*3.1415)/traj_std[x], fc='blue', alpha=0.5)
                #ax.add_patch(rect)
                rect = Rectangle( (0.8, 1), 0.2, 0.2, fc='green', alpha = 0.4)
                ax.add_patch(rect)
                rect = Rectangle( (-1, 1), 0.2, 0.2, fc='red', alpha = 0.4)
                ax.add_patch(rect)
                rect = Rectangle( (-0.8, 1), 1.6, 0.2, fc='grey', alpha = 0.4)
                ax.add_patch(rect)
                grid()
                ylim(0,1.2)
                xlim(-2,2)
                fname ='movie/fig%03d.png'%x
                fig.savefig(fname, bbox_inches='tight')
                fig.clf();

                #print traj_xy[x], traj_std[x]

        elif NUM_DIM==2:
            
            for i in range(len(traj_xy[:,0])):
                fig = figure(1)
                ax = fig.add_subplot(111, aspect='equal')

                plot(traj_xy[0:i+1,0], traj_xy[0:i+1,1], 'b--')
                for j in np.arange(max(i+1-5,0),i+1):
                    circle = Circle( (traj_xy[j,0], traj_xy[j,1]), traj_std[j,0], alpha=0.2, fc='blue')
                    ax.add_patch(circle)

                rect = Rectangle( (-2, -2), 4, 4, fc='grey', alpha = 0.3)
                ax.add_patch(rect)
                rect = Rectangle( (0.8, 0.8), 1.2, 1.2, fc='white', alpha = 0.3)
                ax.add_patch(rect)
                rect = Rectangle( (-1, -1), 0.4, 0.4, fc='red', alpha = 0.3)
                ax.add_patch(rect)
                xlim(-2,2)
                ylim(-2,2)
                fname ='movie/fig%03d.png'%i
                fig.savefig(fname, bbox_inches='tight')
                fig.clf();
                       
        os.system("mencoder 'mf://movie/fig*.png' -mf type=png:fps=1 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o movie/animation.avi")

    fp.close()

if __name__ == "__main__":
    
    read_state_index()
    #read_lqg_trajectories()
    
    if 0:
        read_state_trajectories()
        #draw_goal()
        if(nf1 != "none"):
            fig.savefig(nf1, bbox_inches='tight')
        if(nf2 != "none"):
            fig.savefig(nf2, bbox_inches='tight')
        if( (nf1 == "none") and (nf2 =="none")):
            show()
    else:
        read_belief_traj()
