#!/usr/bin/python

from sys import *
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolor

times = []
NUM_DIM = 2

if len(argv) > 1:
    save_name = argv[1]
else:
    save_name = "none"


def draw_obstacles():
    fig = figure(2)
    axplot = fig.add_subplot(111)
    rect = Rectangle( (0.56, 0.4), 0.14, 0.2, fc='blue', alpha = 0.2)
    axplot.add_patch(rect)

def draw_edges():
    draw_edges = 0
    if draw_edges:
        #t1, x1, t2, x2, prob, delt
        rrg = open("rrg.dat", 'r')
        if rrg:
            lines = rrg.readlines()
            for l in lines:
                s = l.split('\t')
                tx = [float(s[0]),float(s[2])]
                ty = [float(s[1]), float(s[3])]
                if float(s[4]) > 2.0:
                    tmp_alpha = 1.0
                else:
                    tmp_alpha = float(s[4])/2.0
                plot(tx, ty, 'k-', alpha = tmp_alpha, lw=0.5 )

        rrg.close()

def plot_graph(fname):
    
    global times
    rrgp = []
    prob = []
    rrgpf = open("rrgp.dat", 'r')
    if rrgpf:
        lines = rrgpf.readlines()
        for l in lines:
            s = l.split('\t')
            if ( float(s[NUM_DIM]) > -1):
                to_put = [ float(s[i]) for i in range(NUM_DIM) ]
                rrgp.append( to_put )
                prob.append( float(s[NUM_DIM]) )
        
    rrgpf.close()
    
    rrgp = np.array (rrgp)
    prob = np.array(prob)
    
    """ 
    figure(1)    
    if( len(rrgp) > 0):
        for i in range(len(times)):
            for j in range(NUM_DIM):
                
                tmp = [times[i] for x in range(len(rrgp))]
                tmp = array(tmp)
                subplot(NUM_DIM,1,j+1, aspect='auto')
                plot(tmp, rrgp[:,j], 'yo', ms=5.0, alpha = 0.1 )
                grid()
    """
    
    fig = figure(2)
    ax = fig.add_subplot(111, aspect=1.0)
    """
    for i in range(len(rrgp[:,0])):
        circle = Circle( (rrgp[i,0], rrgp[i,1]), 0.005, fc='blue', alpha = 500*prob[i])
        ax.add_patch(circle)
    """
    scatter( rrgp[:,0], rrgp[:,1], marker='o', c= prob[:], s=30, alpha=0.8)
    grid()
    axis([-2, 2, -2, 2])
    
    fig.savefig(fname)

def plot_trajs():
    
    global times
    traj = open("traj.dat", 'r')

    if traj:
        lines = traj.readlines()
        which = 0

        sys = []
        obs = []
        bp = []
        kf = []
        pf = []
        kf_covar = []

        tsys = []
        tobs = []
        tbp = []
        tkf = []
        tpf = []
        tkf_covar = []

        for l in lines:
            s= l.split('\t')
            if len(s) == 1:
                if s[0] == "system\n":
                    which = 0
                elif s[0] == "observation\n":
                    which = 1
                elif s[0] == "best_path\n":
                    which = 2
                elif s[0] == "kf_path\n":
                    which = 3
                elif s[0] == "pf_path\n":
                    which = 4
                elif s[0] == "kf_covar\n":
                    which = 5

            if len(s) > 1:
                time = float(s[0])
                to_put = [ float(s[i+1]) for i in range(NUM_DIM) ]
                if which == 0:
                    tsys.append(time)
                    sys.append(to_put)
                elif which == 1:
                    tobs.append(time)
                    obs.append(to_put)
                elif which == 2:
                    tbp.append(time)
                    bp.append(to_put)
                elif which == 3:
                    tkf.append(time)
                    kf.append(to_put)
                elif which == 4:
                    tpf.append(time)
                    pf.append(to_put)
                elif which == 5:
                    tkf_covar.append(time)
                    kf_covar.append(to_put)

    traj.close()

    sys = array(sys)
    obs = array(obs)
    bp = array(bp)
    kf = array(kf)
    pf = array(pf)
    kf_covar = array(kf_covar)
    
    tsys = array(tsys)
    tobs = array(tobs)
    tbp = array(tbp)
    tkf = array(tkf)
    tpf = array(tpf)
    tkf_covar = array(tkf_covar)
   
    times = tobs

    figure(1)    
    for i in range(NUM_DIM):
        
        ax1 = subplot(NUM_DIM,1,i+1, aspect='auto')
        grid()

        if len(sys) != 0:
            plot( tsys[:], sys[:,i], 'r-', label='sys', lw=1.5)
        #if len(obs) != 0:
            #plot( tobs[:], obs[:,i], 'bx', label='obs')
        
        if len(bp) != 0:
            plot( tbp[:], bp[:,i], 'g-', label='hmm', lw=1.5)
        if len(kf) != 0:
            plot( tkf[:], kf[:,i], 'c-', label='kf', lw=1.5)
        if len(pf) != 0:
            plot( tpf[:], pf[:,i], 'y-', label='pf', lw=1.5)
        
        """
        ax2 = ax1.twinx()
        if len(kf_covar) != 0:
            upper = [ kf[x,i] + sqrt(kf_covar[x,i]) for x in range(len(kf_covar))]
            lower = [ kf[x,i] - sqrt(kf_covar[x,i]) for x in range(len(kf_covar))]
            
            print "kf: ", kf
            print "kf_covar: ", kf_covar
            print "upper: ", upper

            plot( tkf_covar[:], upper, 'm-', label='kf_covar1', lw=1.5)
            plot( tkf_covar[:], lower, 'm-', label='kf_covar2', lw=1.5)
        """
    """
    figure(2)
    if len(sys) != 0:
        plot( sys[:,0], sys[:,1], 'r-', label='sys', lw=1.5)
    if len(bp) != 0:
        plot( bp[:,0], bp[:,1], 'g-', label='sys', lw=1.5)
    if len(kf) != 0:
        plot( kf[:,0], kf[:,1], 'c-', label='kf', lw=1.5)
    #if len(obs) != 0:
        #plot( obs[:,0], obs[:,1], 'bx', label='obs')
    grid()
    """

def plot_sim_trajs():

    mc = open("monte_carlo.dat", 'r')
    
    probs = []
    curr_traj = []
    curr_times = []

    if mc:
        lines = mc.readlines()
        last_prob = 1
        curr_prob = 1
        for l in lines:
            s = l.split('\t')

            if(len(s) ==3):
                last_prob = curr_prob
                curr_prob = 50*float(s[1])
                to_plot = array(curr_traj)
                to_plot_time = array(curr_times)

                if( len(to_plot) > 0):
                    
                    figure(1)    
                    for i in range(NUM_DIM):
                        subplot(NUM_DIM,1,i+1, aspect='auto')
                        grid()
                        plot(to_plot_time[:], to_plot[:,i], 'm-', alpha=0.1)
                   
                    """
                    figure(2)
                    plot(to_plot[:,0], to_plot[:,1], 'mo-', alpha=0.1)
                    grid()
                    """

                curr_traj = []
                curr_times = []
            elif(len(s) == 6):
                curr_times.append( float(s[0]) )
                to_put = [float(s[i+1]) for i in range(NUM_DIM)]
                curr_traj.append( to_put )

        mc.close()

def do_timing_plot():

    ep = open("err_vanderpol.txt", "r")
    tp = open("timing_inc.dat", "r")

    times=[]
    vert=[]
    bpe = []
    kfe=[]

    if ep:
        lines = ep.readlines()
        for l in lines:
            s = l.split('\t')

            vert.append( float(s[0]))
            times.append( float(s[1]))
            bpe.append( float(s[2]))
            kfe.append( float(s[3]))

        vert = array(vert)
        times = array(times)
        bpe = array(bpe)
        kfe = array(kfe)

        figure(4)
        grid()
        plot(vert[:], bpe[:], 'g-', lw=1.5)
        plot(vert[:], kfe[:], 'r-', lw=1.5)
    
    times = []
    vert = []
    
    if tp:
        lines = tp.readlines()
        for l in lines:
            s = l.split('\t')
            
            n = float(s[0])
            vert.append(n)
            times.append( float(s[1])/n/pow(log(n),4))

        vert = array(vert)
        times = array(times)
        
        figure(3)
        grid()
        plot(vert[:], times[:], 'b-', lw=1.5)

def plot_density(fname):
    
    denp = []
    denf = open("data/density.dat", 'r')
    if denf:
        lines = denf.readlines()
        for l in lines:
            s = l.split('\t')
            to_put = [ float(s[i]) for i in range(NUM_DIM+1) ]
            denp.append( to_put )
        
    denf.close()
    
    denp = np.array (denp)
    minp = min(denp[:,0])
    maxp = max(denp[:,0])
    size = [ (3+(x-minp)/(maxp-minp)*1000) for x in denp[:,0] ]
    
    """
    fig = figure(2)
    ax = fig.add_subplot(111, aspect=1.0)
    scatter( denp[:,1], denp[:,2], marker='o', c='b', s= 25, alpha=0.7)
    axis([0, 1, 0, 1])
    grid()

    fig.savefig(fname)
    """
    
    fig = figure(2)
    hexbin(denp[:,1], denp[:,2], bins='log', cmap=cm.get_cmap('Jet'), alpha=0.9, mincnt=1)
    axis([-0.5, 1, -0.5, 1])
    #colorbar()
    fig.savefig(fname)

if __name__ == "__main__":

    plot_trajs()
    #plot_sim_trajs()
    #draw_obstacles()    
    
    #do_timing_plot()

    #plot_graph(save_name)
    #plot_density(save_name)

    #legend()
    
    """
    if save_name != "none":
        fig.savefig(save_name)
    """

    show()

