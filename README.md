# The Replication and Extension of "DuETT: Dual Event Time Transformer for Electronic Health Records"

Credits goes to original DuETT paper author.

Added MIMIC-IV Task training & eval code for reproducing the results:

- `mimic_mortality.py` - Data module for MIMIC IV 2.0 and specifically for ICU Mortality task.
- `train_mimiciv.py` - To run both pretraining and fine-tuning on MIMIV IV 2.0 (ICU Mortality Task), run `python train_mimiciv.py`.
- `dlh_TEAM73_UIN678314629.ipynb` - Jupyter Notebook for end-to-end replication of results.

NOTE: `duett.py` in this repo hasn't been changed vs original paper, but adjustment of hyper-parameters is needed before running the code. Please read carefully about the jupyter notebook above. 

Below is from original author of DuETT paper:

- MIMIC-IV patient ID train and test splits are provided in `mimic-iv-patient-split.json` and were produced using a random shuffle of patient IDs from `hosp/patients.csv`. For our validation split, we use the 15% of patient IDs that appear first in the train split. For PhysioNet, we use the `torchtime` library to load splits.

- To run both pretraining and fine-tuning on PhysioNet-2012, run `python train.py`. Our results were generated with PyTorch 1.13.1, PyTorch Lightning 1.6.1, CUDA 11.7.1, x-transformers 1.5.3, and torchtime 0.5.1.

- The data format used by the model has instances of the form `(x,y)`, where `x` is a tuple `(x_ts, x_static, times)`. `x_ts` is a matrix of size `n_timesteps x d_time_series*2` already in the binned format specified in the paper, with `x_ts[:,:d_time_series]` containing the zero-imputed binned time series values and `x_ts[:,d_time_series:]` containing the corresponding numbers of observations. `x_static` is a vector of all static variables and `times` is a vector of bin end times in (fractional) days.

Note that the `forward` function of the model doesn't take this representation directly, since further preprocessing is required depending on whether the model is being pretrained or fine-tuned. See the model's `training_step` method for details.

