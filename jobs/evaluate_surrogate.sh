python -m surrogate_model.evaluate --checkpoint "surrogate_model/runs/20260525_220128/surrogate_best.pt" --data_path  "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl" --n_examples 2



python -m surrogate_model.evaluate --checkpoint "surrogate_model/runs/20260528_181943/surrogate_best.pt" --data_path  "Data/Dataset:layer_12_stepsize_10_samples_3000_150_400_lc.pkl" --n_examples 2


python -m surrogate_model_latent.evaluate     --checkpoint surrogate_model_latent/runs/20260528_182259/latent_best.pt     --data_path  "Data/Dataset:layer_12_stepsize_10_samples_3000_150_400_lc.pkl"


python -m surrogate_model_latent.evaluate     --checkpoint surrogate_model_latent/runs/20260528_182014/latent_best.pt     --data_path  "Data/Dataset:layer_12_stepsize_10_samples_5000_150_400.pkl"

python -m surrogate_model_latent.evaluate     --checkpoint surrogate_model_latent/runs/20260602_125822/latent_best.pt     --data_path  "Data/Dataset:layer_12_stepsize_10_samples_500_150_400_lc.pkl"

