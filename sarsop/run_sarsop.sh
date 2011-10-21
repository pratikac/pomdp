#!/bin/bash

if [ "$1" = "convert" ]
then
    ./pomdpconvert singleint.pomdp
elif [ "$1" = "solve" ]
then
    ./pomdpsol -p 1e-4 -o singleint.policy singleint.pomdpx
elif [ "$1" = "sim" ]
then
    ./pomdpsim --simLen=100 --simNum=100 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "eval" ]
then
    ./pomdpeval --simLen=100 --simNum=100 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "graph" ]
then
    ./polgraph --policy-file singleint.policy --policy-graph singleint.dot singleint.pomdpx
else
    ./pomdpconvert singleint.pomdp
    ./pomdpsol -p 1e-4 -o singleint.policy singleint.pomdpx
fi
