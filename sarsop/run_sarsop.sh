#!/bin/bash

if [ "$1" = "convert" ]
then
    ../../appl-0.95/src/pomdpconvert singleint.pomdp
elif [ "$1" = "solve" ]
then
    ../../appl-0.95/src/pomdpsol -p 1e-4 -o singleint.policy singleint.pomdpx
elif [ "$1" = "sim" ]
then
    ../../appl-0.95/src/pomdpsim --simLen=100 --simNum=1 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "eval" ]
then
    ../../appl-0.95/src/pomdpeval --simLen=100 --simNum=100 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "graph" ]
then
    ../../appl-0.95/src/polgraph --policy-file singleint.policy --policy-graph singleint.dot singleint.pomdpx
else
    ../../appl-0.95/src/pomdpconvert singleint.pomdp
    ../../appl-0.95/src/pomdpsol -p 1e-3 -o singleint.policy singleint.pomdpx
fi
