mkdir -p checkpoints/results/P

for K in 10 25 50 75 100 125; do
    python test.py --mode=P --K=$K \
        > checkpoints/results/P/result_P_K${K}.txt
done