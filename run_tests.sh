#!/bin/bash
# This script will run all the python files in test and PIapprox directory

curr_dir=$(pwd)
echo "current directory:" $curr_dir


for f in $curr_dir"/tests/optimal_approx/"*.py; do
    if [[ $f != *"/__init__.py" ]]; then
        filename=$(basename "$f")
        name="${filename%.*}"
        echo "Calling " $filename
        python $f
        echo "-----------------------------"
    fi
done