#!/bin/bash

if [ "$1" = "convert" ]
then
    ../../appl-0.95/src/pomdpconvert singleint.pomdp
elif [ "$1" = "solve" ]
then
    ../../appl-0.95/src/pomdpsol -p 1e-4 -o singleint.policy singleint.pomdpx
elif [ "$1" = "sim" ]
then
    ../../appl-0.95/src/pomdpsim --simLen=100 --simNum=200 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "eval" ]
then
    ../../appl-0.95/src/pomdpeval --simLen=100 --simNum=1 --policy-file singleint.policy singleint.pomdpx
elif [ "$1" = "graph" ]
then
    ../../appl-0.95/src/polgraph --policy-file singleint.policy --policy-graph singleint.dot singleint.pomdpx
elif [ "$1" = "draw" ]
then
    for i in {1..50}
    do
        echo $i
        curr_name=`printf "movie/f%03d.png" $i`
        ../../appl-0.95/src/pomdpsim --simLen=100 --simNum=1 --policy-file singleint.policy singleint.pomdpx
        ./sim_analyse.py $curr_name
    done
else
    ../../appl-0.95/src/pomdpconvert singleint.pomdp
    ../../appl-0.95/src/pomdpsol -p 1e-3 -o singleint.policy singleint.pomdpx
    ../../appl-0.95/src/pomdpsim --simLen=100 --simNum=1 --policy-file singleint.policy singleint.pomdpx
fi
