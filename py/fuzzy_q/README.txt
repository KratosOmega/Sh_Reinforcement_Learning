################################ requirements
tensorflow==1.14.0rc1
python==3.7.0
----------------------------
pip install tqdm gym[all]
################################

# train
python3 main.py --env_name=Vissim --is_train=True





################################
#####    >>> INDEX <<<     #####
################################
* origin/naf_v1.4.0_no_training: normal traffic without NN
* origin/naf_v1.4: current main dev branch for branching off
* origin/naf_v1.4.5.2_hidden_300_300_sw : hl=(300, 300) with shock wave penalty
* origin/naf_v1.4.5.3_hidden_30_30_sw : hl=(30, 30) with shock wave penalty
* origin/naf_v1.4.6.1_hidden_30_30 : hl=(30, 30) WITHOUT shock wave penalty
* origin/naf_v1.4.6.2_hidden_300_300 : hl=(300, 300) WITHOUT shock wave penalty
* origin/naf_v1.4.6.3_hidden_60_60 : hl=(60, 60) WITHOUT shock wave penalty
* origin/naf_v1.4.6.4_hidden_15_15 : hl=(15, 15) WITHOUT shock wave penalty
* origin/naf_v1.4.6.1_hidden_30_30_volume_8000 : hl=(30, 30) WITHOUT shock wave penalty
* origin/naf_v1.5.1.1_hidden_30_30: hl=(30, 30), test on random process at episode lvl


################################
#####     >>> TODO <<<     #####
################################

* Update Vissim for environment