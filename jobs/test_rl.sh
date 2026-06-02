


# lc: cooling time = 0.15
python test.py --mode OnlineRL --checkpoint online_RL/runs/20260528_181943/dqn_best.pt 

python -m online_RL.evaluate --checkpoint online_RL/runs/20260528_181943/dqn_best.pt


# 0.1
python -m online_RL.evaluate --checkpoint online_RL/runs/20260601_165727/dqn_best.pt
python test.py --mode OnlineRL --checkpoint online_RL/runs/20260601_165727/dqn_best.pt
python test.py --mode OnlineRL --checkpoint online_RL/runs/20260601_171549/dqn_best.pt  

python test.py --mode RL --checkpoint checkpoints/qnet_offline_3000_10_lc.pt

# 0.15
python test.py --mode OnlineRL --checkpoint online_RL/runs/20260601_234854/dqn_best.pt --surrogate surrogate_model_latent/runs/20260601_162903/latent_best.pt --cool_time 0.15 --results_out online_RL/runs/20260601_234854/matlab_eval.json