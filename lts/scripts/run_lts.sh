#!/bin/bash
# export PATH=/home/luban/environments/openmpi/bin:$PATH
# export LD_LIBRARY_PATH=/home/luban/environments/openmpi/lib:$LD_LIBRARY_PATH
chmod a+x  /nfs/project/chenxionghui/.bashrc
. /nfs/project/chenxionghui/.bashrc
# source /nfs/project/chenxionghui/.bashrc
echo "---- env bashrc cli---"
# conda activate ddrl
python -V
conda info --env
echo "python ../src/run_lts.py $*"

python ../src/run_lts.py $*
