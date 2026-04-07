

python test.py --mode=Constant --lp_const=150.0 > checkpoints/results/constant/result_constant_150.txt
python test.py --mode=Constant --lp_const=200.0 > checkpoints/results/constant/result_constant_200.txt
python test.py --mode=Constant --lp_const=250.0 > checkpoints/results/constant/result_constant_250.txt
python test.py --mode=Constant --lp_const=300.0 > checkpoints/results/constant/result_constant_300.txt

mkdir -p checkpoints/results/P

for K in 10 25 50 75 100 125; do
    python test.py --mode=P --K=$K \
        > checkpoints/results/P/result_P_K${K}.txt
done

python test.py --mode=RL > checkpoints/results/result_5000_10.txt