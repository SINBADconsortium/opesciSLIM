# setup PYTHONPATH for JULIA, you're python path should already contain devito
DEVITO=$PWD
export PYTHONPATH=$PYTHONPATH:$DEVITO/Python/operators/acoustic
export PYTHONPATH=$PYTHONPATH:$DEVITO/Python/operators/tti
export PYTHONPATH=$PYTHONPATH:$DEVITO/Python/containers
# Link librairies
# Compiler
# Load any compiler you want
# e.g module load GCC

# Setup arch
# export DEVITO_ARCH=gnu (GCC)
# export DEVITO_ARCH=clang (mandatory for OSX)
# export DEVITO_ARCH=intel (icc)
# export DEVITO_ARCH=intel-mic (MIc, untested)
# export DEVITO_ARCH=intel-knl (KNL, untested)
# Enable multithread
export DEVITO_OPENMP=1
# Choose number of threads (per julia worker)
export OMP_NUM_THREADS=5
# Enforce affinity, only work for one worker per node
# export KMP_AFFINITY=explicit,proclist=[0,1,2,3,4]
