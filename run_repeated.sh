#!/bin/bash

input="c"
how_many=25
for i in {1..4}
    do
        echo $i
        vert=$(($i*$how_many))
        ./main $vert
        cd sarsop
        ./run_sarsop.sh convert
        ./run_sarsop.sh solve
        cd - 
        echo "finished one loop, press key to continue"
        read input
    done

