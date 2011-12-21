#!/usr/bin/python

import os, sys
from argparse import *
from numpy import *

parser = ArgumentParser(description='Runs sarsop')
parser.add_argument('-i', '--ifile')
parser.set_defaults(ifile='problem')

parser.add_argument('-c', '--convert', action='store_true', dest='convert')
parser.add_argument('-sol', '--solve', action='store_true', dest='solve')
parser.add_argument('-to', '--timeout', default='100')


parser.add_argument('-e', '--eval', action='store_true', dest='eval')
parser.add_argument('-g', '--graph', action='store_true', dest='graph')
parser.add_argument('-t', '--test', action='store_true', dest='test')

parser.add_argument('-sim', '--sim', action='store_true', dest='sim')
parser.add_argument('-l', '--simlen', default='200')
parser.add_argument('-n', '--simnum', default='1000')
parser.add_argument('-b', '--write-belief', action='store_true', dest='write_belief')


args = parser.parse_args()

prob_file = args.ifile+'.pomdp'
pomdp_file=args.ifile+'.pomdpx'
policy_file=args.ifile+'.policy'
graph_file=args.ifile+'.dot'

if args.convert:
    os.system("src/pomdpconvert "+prob_file)
if args.solve:
    os.system("src/pomdpsol --timeout "+args.timeout+" -p 1e-1 -o "+policy_file+" "+prob_file)
if args.sim and (not args.write_belief):
    os.system("src/pomdpsim --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file)
if args.sim and args.write_belief:
    os.system("src/pomdpsim --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file+' --write-belief true')
if args.eval:
    os.system("src/pomdpeval --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file)
if args.graph:
    os.system("src/polgraph --policy-file "+policy_file+" --policy-graph "+graph_file+" "+prob_file)
    
if args.test:
    num_vert = [10,11,12,13,14,15]
    discount = [0.98]
    process_noise = 0.01*linspace(1,1,1)
    for vert in num_vert:
        for d in discount:
            for noise in process_noise:
                os.system("rm "+prob_file+" "+policy_file)
                os.system("cd ..;./main "+str(vert)+" "+str(d)+" "+str(noise)+";cd -")
                os.system("src/pomdpsol --timeout "+args.timeout+" -p 1e-1 -o "+policy_file+" "+prob_file)
                os.system("src/pomdpsim --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file)
                os.system("./sim_analyse.py figures/t1.png figures/fig_"+str(vert)+"_"+str(d)+"_"+str(noise)+".png")
                #raw_input()
