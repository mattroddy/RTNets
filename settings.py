import argparse
import copy
import json
import os
import pdb
import pickle
import platform
import time as t
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch

from dataset_settings import (data_dir, dev_saved_data_fold,
                              fbanks_50ms_dict_list, full_saved_data_fold,
                              gemaps_no_mfccs_50ms_dict_list)
from experiment_paths import (encodings_folder, just_test_folder,
                              second_encodings_folder)
from util import get_data_loader_settings, get_fold_name

feature_dict_list = fbanks_50ms_dict_list + gemaps_no_mfccs_50ms_dict_list


parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size', default=128, type=int
)

parser.add_argument(
    '--just_test', action='store_true'
)

parser.add_argument(
    '--full_test', action='store_true'
)

parser.add_argument(
    '--fixed_test', action='store_true'
)

parser.add_argument(
    '--just_test_folder', default='', type=str
)
parser.add_argument(
    '--max_strt_test', default=2.0, type=float
)

parser.add_argument(
    '--max_wait_train', default=15.0, type=float
)

parser.add_argument(
    '--test_valid', action='store_true'
)
parser.add_argument(
    '--l2', default=1e-05, type=float
)
parser.add_argument(
    '--enc_abl', default='none', type=str, help='encoder ablation settings: none, no_enc, only_ling, only_acous'
)
parser.add_argument(
    '--inf_abl', default='none', type=str, help='inference network ablation: none, only_ling, only_acous'
)
parser.add_argument(
    '--use_vae', action='store_true'
)
parser.add_argument(
    '--w_kl', default=0.0, type=float
)
parser.add_argument(
    '--note_append', default='', type=str
)

parser.add_argument(
    '--seed', default=1, type=int
)

args = parser.parse_args()

if platform.node() == 'matt-xps':
    # DEV SETTINGS
    use_saved_data_bool = False
    use_saved_data_fold = data_dir+'/saved_datasets/dev1/'
    complete_file_list = ['sw2005']
    train_list_path = './splits/ms_state_no_nxt.txt'
    train_file_list = complete_file_list
    valid_file_list = complete_file_list
    test_file_list = complete_file_list
    update_data_path_train = data_dir+'/update_data_ms_state_full_200ms.p'
    update_data_path_test = data_dir+'/update_data_200ms.p'
    batch_size = 2
    pad_noise_bool = False
    note_append = '_dev'

elif platform.node() == 'matt-pc':
    # elif True:
    # DEV SETTINGS
    use_saved_data_bool = False
    use_saved_data_fold = data_dir+'/saved_datasets/dev1/'
    train_list_path = './splits/ms_state_no_nxt.txt'
    nxt_files_path = './splits/complete_nxt.txt'
    nxt_file_list = list(pd.read_csv(
        nxt_files_path, header=None, dtype=str)[0])
    update_data_path_train = data_dir+'/update_data_ms_state_full_200ms.p'
    update_data_path_test = data_dir+'/update_data_200ms.p'
    train_file_list = nxt_file_list[:4]
    valid_file_list = nxt_file_list[4:8]
    test_file_list = nxt_file_list[4:8]
    batch_size = 4
    pad_noise_bool = True
    note_append = '_dev'

else:
    # REAL SETTINGS -- With Noise
    use_saved_data_bool = True
    use_saved_data_fold = full_saved_data_fold
    batch_size = args.batch_size
    train_list_path = './splits/ms_state_no_nxt.txt'
    ms_state_no_nxt = list(pd.read_csv(
        train_list_path, header=None, dtype=str)[0])
    # train_file_list = ms_state_no_nxt[:-64]
    train_file_list = ms_state_no_nxt[:-150]
    nxt_files_path = './splits/complete_nxt.txt'
    nxt_file_list = list(pd.read_csv(
        nxt_files_path, header=None, dtype=str)[0])
    update_data_path_train = data_dir+'/update_data_ms_state_full_200ms.p'
    update_data_path_test = data_dir+'/update_data_200ms.p'
    # valid_file_list = ms_state_no_nxt[-64:]
    valid_file_list = ms_state_no_nxt[-150:]
    test_file_list = nxt_file_list
    pad_noise_bool = True
    note_append = args.note_append
    # import sys
    # if len(sys.argv) >= 2:
    #   note_append = str(sys.argv[1])
    # else:
    #   note_append = ''+input("Enter note_append:")


# GENERAL SETTINGS
use_ling = True
num_data_loader_workers = 0
out_str = 'NT'
use_wait_stop_tok = True
language_size = 10000  # options: 500, 5000, 10000, 20000, 30080
ling_use_glove = True
ling_emb_freeze = False
ling_timings = {
    'updates': 'held',
    'response': 'held',
    'inference': 'one_shot_unspec_held',
    # 'inference':'one_shot'
}

max_epochs = 20  # max num epochs
optim_patience = 10
framelen = 0.05
pad_all_max_len_bool = False
# 3478 is the maximum test sequence length. We append 5 seconds after that.
pad_max_len = 3478 + 100

## Test settings ##
load_model = False
just_test = args.just_test
test_best_model = False
vae_experiments = False
vae_target_da = 'ny'  # ar, aa, 'nn', 'ny'
vae_data_multiplier = 1000  # 20, 1, 4, 2
vae_target_second_da = 'nn'
vae_data_multiplier_2 = 2
load_encodings = False
load_second_encodings = False
# use vae_data_multiplier to control the number of each turnpair
individual_turnpair_experiments = False
# sig_offsets = pd.read_csv('./significant_offsets.csv')
# # tp_files, tp_abs, tp_sys_trns = [test_file_list[0]], ['A'], [4]
# sig_i = 5
# tp_files, tp_abs, tp_sys_trns = [sig_offsets.file[sig_i]], [
#     sig_offsets.a_usr[sig_i]], [int(sig_offsets.sys_trn_idx[sig_i])]
# target_individual_turnpairs = [tp_files, tp_abs, tp_sys_trns]
test_valid = args.test_valid
sanity_check_bool = False
sanity_check_file = [test_file_list[0]]
test_autoregress = False
# full_test_flag = True if just_test else False
full_test_flag = args.full_test
# maximum time before the system's ground truth
# start of utterance that the system is active (in seconds)
# Set to -1 for no maximum
# Only used when sampling. Other tests use max_wait_train
max_strt_test = args.max_strt_test  # 2.0
plot_batch_bool = False
use_fixed_test_prob = args.fixed_test
# fixed_test_prob = 0.0579  # 0.1179, 0.0773, 0.0648, 0.0597, 0.0579, max=roughly 0.03
fixed_test_prob = 0.02881  # 0.1179, 0.0773, 0.0648, 0.0597, 0.0579
fixed_test_logit = -1 * np.log(1.0 / fixed_test_prob - 1)
extra_pad_frames = 80
# 200 was 80 # note: time_out only used in stats calculation during full_test. Doesn't affect training.
time_out_length = 200
# Training temperature is hard coded to be 1.0 (see run_cont.py)
temperature = 1.0
analyze_error = False

# just_test_model = '/model.pt'
just_test_model = '/best_model.pt' if test_best_model else '/model.pt'
if load_model and not just_test:
    just_test_model = '/model.pt'
encoder_settings = {
    'skip_vae': not args.use_vae,
    # note this will override previous use_ling (but only for encoder)
    'use_ling': True,
    'use_acous': True,
}
just_test_folder = args.just_test_folder if not (
    args.just_test_folder == '') else just_test_folder
two_sys_turn = False  # This only applies to training
lstm_sets_dict = {
    'just_test_folder': just_test_folder,
    'just_test_model': just_test_model,
    'test_best_model': test_best_model,
    'test_autoregress': test_autoregress,
    'full_test_flag': full_test_flag,
    'test_valid': test_valid,
    'analyze_error': analyze_error,
    'plot_batch': plot_batch_bool,
    # 'none','no_enc','only_ling','only_acous'
    'enc_ablation_setting': args.enc_abl,
    # 'none','only_ling','only_acous','no_context'
    'dec_ablation_setting': args.inf_abl,
    'train_random_sample': True,
    'temperature': temperature,
    # 'max_pre_encoding': max_pre_encoding,
    'two_sys_turn': two_sys_turn,
    'response_encoder_hidden_size': 256,  # 64, 256
    'master_encoder_hidden_size': 256,
    'inference_hidden_size': 1024,  # 128
    'vae_dim': 4,  # was 128
    'enc_reduce_size': 256,
    'full_over': True,  # if False will omit datapoints where there is no associated user turn
    # maximum time before the system's ground truth start of utterance that the system is active (in seconds) Set to -1 for no maximum
    'max_strt_test': max_strt_test,
    # 15  # maximum amount of wait tokens (in seconds) before the system's first utterance (only in train). Set to -1 for no maximum
    'max_wait_train': args.max_wait_train if not (just_test and full_test_flag) else max_strt_test,
    # use long version, where the maximum amount of wait tokens before the system's utterance isn't constrained by the previous usr ipu
    'max_wait_train_from_usr_turn': False,
    'random_strt_train': True,
    'embeds_dropout': 0.0,
    'l2': args.l2,  # 1e-05, 1e-07
    'learning_rate': 0.0005,  # 0.0002
    # One milestone is 200 iterations e.g. milestone 75 is iteration 15000
    # last milestone is when training ends.
    # 9000, 11000, 13000, 14000, 15000
    'milestones': [45, 55, 65, 70, 75],  # [25,35,45,50]
    'pad_all_max_len_bool': pad_all_max_len_bool,
    'pad_max_len': pad_max_len,
    'extra_pad_frames': extra_pad_frames,
    'time_out_length': time_out_length,  # 5 second timeout
    'train_autoregress': False,
    'valid_autoregress': False,
    'valid_full_test_flag': False,  # True
    'test_full_test_flag': False,
    'test_end_epoch': False,
    'encoder_settings': encoder_settings,
    'use_ling': use_ling,
    'language_size': language_size,
    'ling_timings': ling_timings,
    'use_wait_stop_tok': use_wait_stop_tok,
    'ling_use_glove': ling_use_glove,
    'ling_emb_freeze': ling_emb_freeze,
    'pad_noise_bool': pad_noise_bool,
    'just_test': just_test,
    'sanity_check_bool': sanity_check_bool,
    'sanity_check_file': sanity_check_file,
    'vae_experiments': vae_experiments,
    'vae_target_da': vae_target_da,
    'vae_target_second_da': vae_target_second_da,
    'vae_data_multiplier': vae_data_multiplier,
    'individual_turnpair_experiments': individual_turnpair_experiments,
    # 'target_individual_turpairs': target_individual_turnpairs,
    'load_encodings': load_encodings,
    'load_second_encodings': load_second_encodings,
    'encodings_folder': encodings_folder,
    'second_encodings_folder': second_encodings_folder,
    'use_fixed_test_prob': use_fixed_test_prob,
    'fixed_test_prob': fixed_test_prob,
    'fixed_test_logit': fixed_test_logit,
    'seed': args.seed
}
lstm_sets_dict['response_encoder_hidden_size'] = 0 if lstm_sets_dict[
    'enc_ablation_setting'] == 'no_enc' else lstm_sets_dict['response_encoder_hidden_size']

# Turn on/off prediction tasks
pred_task_dict = OrderedDict()

pred_task_dict['NT'] = OrderedDict([
    ('bool', True),
    ('weight', 0.0),
    ('pred_len', 1),
    ('output_layers', 0),  # there is an extra linear layer as well
    ('bypass_layers', False),
    ('output_layer_size', 0),
    ('in_dim', lstm_sets_dict['inference_hidden_size'])
])

pred_task_dict['TL'] = OrderedDict([
    ('bool', True),
    ('weight', 1.0),
    ('pred_len', 1),
    ('output_layers', 1),  # there is an extra linear layer as well
    ('bypass_layers', True),
    ('output_layer_size', 32),
    ('in_dim', lstm_sets_dict['inference_hidden_size'])
])

pred_task_dict['KLD'] = OrderedDict([
    ('bool', not encoder_settings['skip_vae']),
    ('weight', args.w_kl),
    ('pred_len', 1),
    ('output_layers', 1),  # there is an extra linear layer as well
    ('bypass_layers', True),
    ('output_layer_size', 32),
    ('in_dim', lstm_sets_dict['inference_hidden_size'])
])


lstm_sets_dict['max_rand_strt_test_f'] = int(
    20 * lstm_sets_dict['max_strt_test'])
lstm_sets_dict['max_wait_train_f'] = int(20*lstm_sets_dict['max_wait_train'])
# set number of frames before end of ipu to use for cont_hold_shift
pred_task_dict['n_pre'] = 2
pred_task_dict['active_outputs'] = [
    key for key, val in pred_task_dict.items()
    if isinstance(val, dict) and val['bool']
]
pred_task_dict['active_output_indices'] = {act: n for n,
                                           act in enumerate(pred_task_dict['active_outputs'])}

lstm_sets_dict['pred_task_dict'] = pred_task_dict

if use_saved_data_bool:
    if not os.path.exists(use_saved_data_fold):
        os.makedirs(use_saved_data_fold)

if platform.node() == 'matt-pc':
    print('USING DEV SETs')
else:
    print('USING FULL SETs')
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")
print('Using: ' + device.type)


if 'Map' in os.getcwd():
    data_set_select = 0
elif 'Mahnob' in os.getcwd():
    data_set_select = 1
elif 'SWB' in os.getcwd():
    data_set_select = 2
data_select_dict = {0: ['f', 'g'], 1: ['c1', 'c2'], 2: ['A', 'B']}
time_label_select_dict = {
    0: 'frame_time',
    1: 'timestamp'
}
data_select_list = data_select_dict[data_set_select]
output_order_train, time_bool_indices, num_feat_per_person = get_data_loader_settings(
    feature_dict_list)

if just_test:
    results_dict = json.load(open(just_test_folder + '/results.json', 'r'))
    note_append += '_best_'+str(lstm_sets_dict['test_best_model'])
    note_append += '_JUST_TEST'
    note_append += '_test_AR_'+str(lstm_sets_dict['test_autoregress'])
    lstm_sets_dict['temperature'] = temperature
else:
    lstm_sets_dict['temperature'] = 1.0  # always train with temperature at 1.0

naming_dict = {}
naming_dict['time_str'] = t.strftime('%Y%m%d%H%M%S')[3:]
naming_dict['lstm_sets_dict'] = lstm_sets_dict
naming_dict['batch_size'] = batch_size
naming_dict['note'] = note_append
naming_dict = get_fold_name(naming_dict, lstm_sets_dict)
if not os.path.exists('./results'):
    os.mkdir('./results')
os.mkdir(naming_dict['fold_name'])
print('\n **** \n')
print(naming_dict['fold_name'])
print('\n **** \n')
json.dump(lstm_sets_dict, open(
    naming_dict['fold_name']+'/lstm_settings_dict.json', 'w'), indent=4)
print('Loading Annots:')

# get update_data
t0 = t.time()
update_annots_train = pickle.load(open(update_data_path_train, 'rb'))
update_annots_test = pickle.load(open(update_data_path_test, 'rb'))
print('time taken: \n'+str(t.time()-t0))


# Train dataloader
train_dataset_settings_dict = {
    'extra_pad_frames': extra_pad_frames,
    'use_saved_data_bool': use_saved_data_bool,
    'use_saved_data_fold': use_saved_data_fold,
    'batch_size': batch_size,
    'pad_noise_bool': pad_noise_bool,
    'output_order': output_order_train,
    'use_ling': use_ling,
    'ling_use_glove': ling_use_glove,
    'ling_emb_freeze': ling_emb_freeze,
    'ling_timings': ling_timings,
    'update_annots': update_annots_train,
    'file_list': train_file_list,
    'set_type': 'train',
    'feature_dict_list': feature_dict_list,
    'pred_task_dict': pred_task_dict,
    'data_select_list': data_select_list,
    'num_feat_per_person': num_feat_per_person,
    'device': device,
    'num_preloader_workers': 0,
    'lstm_sets_dict': lstm_sets_dict,
}

# Valid dataloader
valid_dataset_settings_dict = copy.copy(train_dataset_settings_dict)
valid_dataset_settings_dict['file_list'] = valid_file_list
valid_dataset_settings_dict['update_annots'] = update_annots_train
valid_dataset_settings_dict['update_annots_test'] = update_annots_train
valid_dataset_settings_dict['set_type'] = 'valid'

# test dataloader
test_dataset_settings_dict = copy.copy(train_dataset_settings_dict)
test_dataset_settings_dict['file_list'] = test_file_list
test_dataset_settings_dict['update_annots'] = update_annots_train
test_dataset_settings_dict['update_annots_test'] = update_annots_test
test_dataset_settings_dict['set_type'] = 'test'

if lstm_sets_dict['test_valid']:
    test_file_list = valid_file_list
    test_dataset_settings_dict = valid_dataset_settings_dict

print('Done with settings')
