#!/bin/bash
# This file contains the commands to reproduce the results
# from Table 1. 

# ---- Training commands

# (1) Full Model
python run.py --note_append full_model

# (3) Encoder Ablation: No Encoder
python run.py --note_append no_enc --enc_abl no_enc

# (4) Encoder Ablation: Only Acoustic
python run.py --note_append encAcous --enc_abl only_acous

# (5) Encoder Ablation: Only Linguistic
python run.py --note_append encLing --enc_abl only_ling

# (6) Inference Ablation: Only Acoustic
python run.py --note_append infAcous --inf_abl only_acous

# (7) Inference Ablation: Only Linguistic
python run.py --note_append infLing --inf_abl only_ling

# (8) VAE: w_kl=0.0
python run.py --note_append KL_0 --use_vae --w_kl 0.0

# (9) VAE: w_kl=10^-4
python run.py --note_append KL_1e-04 --use_vae --w_kl 1e-04

# (10) VAE: w_kl=10^-3
python run.py --note_append KL_1e-03 --use_vae --w_kl 1e-03

# (11) VAE: w_kl=10^-2
python run.py --note_append KL_1e-02 --use_vae --w_kl 1e-02

# (12) VAE: w_kl=10^-1
python run.py --note_append KL_1e-01 --use_vae --w_kl 1e-01


# ---- Test commands

# To get the BCE test loss for a model include the 
# relevant setting flags (e.g. --enc_abl...) for the model
# as well as: 
# "--just_test --just_test_fold <path_to_model_folder> --max_strt_test -1.0 --max_wait_train -1.0"
# Setting the last two flags to -1.0 removes a 15 second size limit on the span of R_START. 
# The size limit is used by default during training since it was found to aid with the stability
# of the training loss during early experiments.
#
# To get the mean-absolute offset error (MAE), run the same test commands from above, but with
# the additional flag:
# "--full_test"
# 
# To get the results from row 2 (Fixed probability) use the flag
# "--fixed_test"
