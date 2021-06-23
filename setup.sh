toplevel=`git rev-parse --show-toplevel`
export PYTHONPATH=$PYTHONPATH:$toplevel/
export MKL_NUM_THREADS=1

