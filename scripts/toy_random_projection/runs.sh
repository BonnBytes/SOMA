# RANDOM RUNS
python scripts/toy_random_projection/projection.py --init=r --embed_dim=1024 --rank=32
# FOURIER RUNS
python scripts/toy_random_projection/projection.py --init=f --embed_dim=1024 --ut_lambda=z --rank=32
# python scripts/toy_random_projection/projection.py --init=f --embed_dim=1024 --ut_lambda=r --rank=32
# python scripts/toy_random_projection/projection.py --init=f --embed_dim=1024 --ut_lambda=o --rank=32
# WAVELET RUNS
python scripts/toy_random_projection/projection.py --init=w --embed_dim=1024 --ut_lambda=z --rank=32
# python scripts/toy_random_projection/projection.py --init=w --embed_dim=1024 --ut_lambda=o --rank=32
# python scripts/toy_random_projection/projection.py --init=w --embed_dim=1024 --ut_lambda=r --rank=32
# HADAMARD RUNS
python scripts/toy_random_projection/projection.py --init=h --embed_dim=1024 --ut_lambda=z --rank=32
# python scripts/toy_random_projection/projection.py --init=h --embed_dim=1024 --ut_lambda=o --rank=32
# python scripts/toy_random_projection/projection.py --init=h --embed_dim=1024 --ut_lambda=r --rank=32
# HIPPO RUNS
# python scripts/toy_random_projection/projection.py --init=m --embed_dim=1024 --ut_lambda=z --rank=32
# python scripts/toy_random_projection/projection.py --init=m --embed_dim=1024 --ut_lambda=o --rank=32
# python scripts/toy_random_projection/projection.py --init=m --embed_dim=1024 --ut_lambda=r --rank=32