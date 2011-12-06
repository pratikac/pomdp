#!/usr/bin/python

import os, sys
from argparse import *

parser = ArgumentParser(description='Runs sarsop')
parser.add_argument('-i', '--ifile')
parser.set_defaults(ifile='problem')

parser.add_argument('-c', '--convert', action='store_true', dest='convert')
parser.add_argument('-sol', '--solve', action='store_true', dest='solve')

parser.add_argument('-sim', '--sim', action='store_true', dest='sim')
parser.add_argument('-l', '--simlen', default='100')
parser.add_argument('-n', '--simnum', default='100')
parser.add_argument('-b', '--write-belief', action='store_true', dest='write_belief')


args = parser.parse_args()

prob_file = args.ifile+'.pomdp'
pomdp_file=args.ifile+'.pomdpx'
policy_file=args.ifile+'.policy'

if args.convert:
    os.system("src/pomdpconvert "+prob_file)
if args.solve:
    os.system("src/pomdpsol --randomization -p 1e-2 -o "+policy_file+" "+prob_file)
if args.sim and (not args.write_belief):
    os.system("src/pomdpsim --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file)
if args.sim and args.write_belief:
    os.system("src/pomdpsim --simLen="+args.simlen+" --simNum="+args.simnum+" --policy-file "+policy_file+" "+prob_file+' --write-belief true')


