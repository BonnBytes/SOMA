# RANDOM RUNS
python projection.py --init=r --full_rank=512 --rank=32
# FOURIER RUNS
python projection.py --init=f --full_rank=512 --ut_lambda=z --rank=32
python projection.py --init=f --full_rank=512 --ut_lambda=o --rank=32
python projection.py --init=f --full_rank=512 --ut_lambda=r --rank=32
# WAVELET RUNS
python projection.py --init=w --full_rank=512 --ut_lambda=z --rank=32
python projection.py --init=w --full_rank=512 --ut_lambda=o --rank=32
python projection.py --init=w --full_rank=512 --ut_lambda=r --rank=32
# HADAMARD RUNS
python projection.py --init=h --full_rank=512 --ut_lambda=z --rank=32
python projection.py --init=h --full_rank=512 --ut_lambda=o --rank=32
python projection.py --init=h --full_rank=512 --ut_lambda=r --rank=32
# HIPPO RUNS
python projection.py --init=m --full_rank=512 --ut_lambda=z --rank=32
python projection.py --init=m --full_rank=512 --ut_lambda=o --rank=32
python projection.py --init=m --full_rank=512 --ut_lambda=r --rank=32