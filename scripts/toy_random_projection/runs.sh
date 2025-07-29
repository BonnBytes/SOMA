# RANDOM RUNS
python simple_convergence.py --init=r --full_rank=1024 --rank=32
# FOURIER RUNS
python simple_convergence.py --init=f --full_rank=1024 --untrain_lambda_init=z --rank=32
python simple_convergence.py --init=f --full_rank=1024 --untrain_lambda_init=o --rank=32
python simple_convergence.py --init=f --full_rank=1024 --untrain_lambda_init=r --rank=32
# # WAVELET RUNS
python simple_convergence.py --init=w --full_rank=1024 --untrain_lambda_init=z --rank=32
python simple_convergence.py --init=w --full_rank=1024 --untrain_lambda_init=o --rank=32
python simple_convergence.py --init=w --full_rank=1024 --untrain_lambda_init=r --rank=32
# # HADAMARD RUNS
python simple_convergence.py --init=h --full_rank=1024 --untrain_lambda_init=z --iterations=1000 --rank=32
python simple_convergence.py --init=h --full_rank=1024 --untrain_lambda_init=o --iterations=1000 --rank=32
python simple_convergence.py --init=h --full_rank=1024 --untrain_lambda_init=r --iterations=1000 --rank=32
# # HIPPO RUNS
python simple_convergence.py --init=m --full_rank=1024 --untrain_lambda_init=z --rank=32
python simple_convergence.py --init=m --full_rank=1024 --untrain_lambda_init=o --rank=32
python simple_convergence.py --init=m --full_rank=1024 --untrain_lambda_init=r --rank=32