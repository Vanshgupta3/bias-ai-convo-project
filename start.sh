export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
uvicorn app:app --host 0.0.0.0 --port 10000 --workers 1
