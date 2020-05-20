# from settings import vae_target_da, vae_data_multiplier, vae_experiments
import copy
import gc
import json
import logging
import os
import pdb
import pickle
import time as t
import warnings
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_curve, roc_curve)
from torch import tensor as tn

warnings.filterwarnings(
    "ignore", message="The covariance matrix associated to your dataset is not full rank")


def convert_to_ms_int_floor(x):
    # converts float ms to rounded integers
    # use convention >= and <
    frame_length2 = 0.05 * 100
    return np.int32(np.floor((np.array(x) * 100 / frame_length2)) * frame_length2)


def get_data_loader_settings(feature_dict_list):
    tb_indx = 0
    num_feat_per_person = {'acous': 0}
    output_order_train, time_bool_indices = [], []
    for feat_dict in feature_dict_list:
        num_feat_per_person['acous'] += len(feat_dict['features']) * \
            feat_dict['stack_size']
        if not('x/' + feat_dict['modality'] in output_order_train):
            output_order_train.append('x/' + feat_dict['modality'])
            tb_indx += 1
            if feat_dict['is_irregular']:
                output_order_train.append(
                    'time_bools/' + feat_dict['modality'])
                time_bool_indices.append(tb_indx)
                tb_indx += 1
    # model_in_length = len(output_order_train)
    return output_order_train, time_bool_indices, num_feat_per_person


def get_vae_dataset(dataset, update_annots_test, target_da):
    print('getting new dataset')
    # da = 'aa' #ar
    a_b_switch = {'A': 'B', 'B': 'A'}
    indices = []
    for i, data in enumerate(dataset):
        file, a_usr = data['file'][0], data['a_usr'][0]
        b_sys = a_b_switch[a_usr]
        sys_trn_idx = data['update_idx']
        usr_trn_idx = data['associated_usr_turn']
        if usr_trn_idx == -1:
            continue
        sys_turn_strt_da_swbd = update_annots_test[file][b_sys]['turn_words_start_swbdType'][sys_trn_idx]
        sys_turn_end_da_swbd = update_annots_test[file][b_sys]['turn_words_end_swbdType'][sys_trn_idx]
        sys_turn_full_overlap = update_annots_test[file][b_sys]['turn_full_overlap'][sys_trn_idx]
        sys_turn_one_da_bool = update_annots_test[file][b_sys]['turn_one_da_bool'][sys_trn_idx]
        if sys_turn_strt_da_swbd == target_da:
            indices.append(i)

    return indices


def get_individual_turnpair_dataset(dataset, update_annots_test, files, a_usrs, sys_trn_indxs):
    print('getting new individual dataset')
    a_b_switch = {'A': 'B', 'B': 'A'}
    indices = []
    for i, data in enumerate(dataset):
        file, a_usr = data['file'][0], data['a_usr'][0]
        b_sys = a_b_switch[a_usr]
        sys_trn_idx = data['update_idx']
        usr_trn_idx = data['associated_usr_turn']
        if usr_trn_idx == -1:
            continue
        sys_turn_strt_da_swbd = update_annots_test[file][b_sys]['turn_words_start_swbdType'][sys_trn_idx]
        sys_turn_end_da_swbd = update_annots_test[file][b_sys]['turn_words_end_swbdType'][sys_trn_idx]
        sys_turn_full_overlap = update_annots_test[file][b_sys]['turn_full_overlap'][sys_trn_idx]
        sys_turn_one_da_bool = update_annots_test[file][b_sys]['turn_one_da_bool'][sys_trn_idx]
        if file in files and a_usr in a_usrs and sys_trn_idx in sys_trn_indxs:
            indices.append(i)

    return indices


def get_vae_encodings(lstm_sets_dict, second_folder_flag):
    print('getting mu and log_var')

    fold_str = 'second_encodings_folder' if second_folder_flag else 'encodings_folder'
    mus = pickle.load(open(lstm_sets_dict[fold_str]+'/sys_encode_mu.p', 'rb'))
    log_vars = pickle.load(
        open(lstm_sets_dict[fold_str]+'/sys_encode_log_var.p', 'rb'))
    log_var = np.log(np.mean(np.exp(log_vars), 0))
    mu = np.mean(mus, 0)
    print('done getting mu and log_var')

    return mu, log_var


def get_file(arg):

    # inputs
    filename, dataset_settings_dict = arg
    feature_dict_list = dataset_settings_dict['feature_dict_list']
    data_select_list = dataset_settings_dict['data_select_list']

    # output dicts
    data_f = {'acous': {'x': {}, 'x_i': {}},
              'visual': {'x': {}, 'x_i': {}}}
    data_g = {'acous': {'x': {}, 'x_i': {}},
              'visual': {'x': {}, 'x_i': {}}}
    output_dict = {}
    output_dict['total_embed_in_dim'] = {'acous': 0, 'visual': 0}
    output_dict['total_embed_out_dim'] = {'acous': 0, 'visual': 0}
    output_dict['feature_name_list'] = {'acous': [], 'visual': []}
    output_dict['active_modalities'] = []
    output_dict['uses_master_time_rate_bool'] = {'acous': True, 'visual': True}
    output_dict['time_step_size'] = {'acous': 1, 'visual': 1}
    output_dict['is_irregular'] = {'acous': False, 'visual': False}
    output_dict['dur_multiplier'] = {'acous': False, 'visual': False}

    # is set every file loop, probably some other more efficient way
    output_dict['embedding_info'] = {'acous': [], 'visual': []}
    for feature_dict in feature_dict_list:
        # Get settings for the modality
        if not(feature_dict['modality'] in output_dict['active_modalities']):
            output_dict['active_modalities'].append(feature_dict['modality'])
            output_dict['time_step_size'][feature_dict['modality']
                                          ] = feature_dict['time_step_size']
            output_dict['dur_multiplier'][feature_dict['modality']
                                          ] = feature_dict['dur_multiplier']
            # uses time_bools to change lstm input or not
            output_dict['is_irregular'][feature_dict['modality']
                                        ] = feature_dict['is_irregular']
            # is 50ms or 10ms sampled
            output_dict['uses_master_time_rate_bool'][feature_dict['modality']
                                                      ] = feature_dict['uses_master_time_rate']

        # if not(feature_dict['is_h5_file']):
        data_f_temp = pd.read_csv(
            feature_dict['folder_path']+'/'+filename+'.'+data_select_list[0]+'.csv')
        data_g_temp = pd.read_csv(
            feature_dict['folder_path']+'/'+filename+'.'+data_select_list[1]+'.csv')
        if 'embedding' in feature_dict and feature_dict['embedding'] == True:
            embed_info = {}
            for embed_key in ['features', 'embedding', 'title_string', 'embedding_num', 'embedding_in_dim', 'embedding_out_dim', 'embedding_use_func', 'use_glove', 'glove_embed_table']:
                embed_info[embed_key] = feature_dict[embed_key]
            output_dict['embedding_info'][feature_dict['modality']].append(
                embed_info)
            output_dict['total_embed_in_dim'][feature_dict['modality']
                                              ] = output_dict['total_embed_in_dim'][feature_dict['modality']] + embed_info['embedding_in_dim']
            output_dict['total_embed_out_dim'][feature_dict['modality']
                                               ] = output_dict['total_embed_out_dim'][feature_dict['modality']] + embed_info['embedding_out_dim']
            output_dict['embedding_info'][feature_dict['modality']][-1]['emb_indices'] = [(len(output_dict['feature_name_list'][feature_dict['modality']]), len(
                output_dict['feature_name_list'][feature_dict['modality']])+embed_info['embedding_in_dim'])]
        feat_stack_names = []
        for feature_name in feature_dict['features']:
            stack_size = feature_dict['stack_size'] if 'stack_size' in list(
                feature_dict.keys()) else 1

            for stack_idx in range(stack_size):
                feat_stack_name = feature_name + '_' + str(stack_idx)
                feat_stack_names.append(feat_stack_name)
                end_idx = len(data_f_temp[feature_name]
                              ) if stack_idx == 0 else -stack_idx
                data_f[feature_dict['modality']
                       ]['x'][feat_stack_name] = np.array(
                    np.zeros(stack_idx).tolist()+data_f_temp[feature_name][0:end_idx].tolist(), dtype=np.float32)
                data_g[feature_dict['modality']
                       ]['x'][feature_name + '_' + str(stack_idx)] = np.array(
                    np.zeros(stack_idx).tolist()+data_g_temp[feature_name][0:end_idx].tolist(), dtype=np.float32)
        # this makes other code for name lists superfluous
        # self.feature_name_list[feature_dict['modality']] += feature_dict['features']
        output_dict['feature_name_list'][feature_dict['modality']
                                         ] += feat_stack_names

    output_dict['num_feat_per_person'] = {'acous': len(data_f['acous']['x'].keys()), 'visual': len(
        data_f['visual']['x'].keys())}  # this is half the dimension of the output of dataloader

    data_f_dict, data_g_dict = {}, {}
    data_f_dict[filename], data_g_dict[filename] = {}, {}
    for modality in output_dict['active_modalities']:
        f_min = np.min([len(d) for d in data_f[modality]['x'].values()])
        g_min = np.min([len(d) for d in data_g[modality]['x'].values()])
        len_min = np.min([f_min, g_min])
        data_f_dict[filename][modality] = [
            data_f[modality]['x'][feature_name][:len_min]
            for feature_name in output_dict['feature_name_list'][modality]
        ]
        data_g_dict[filename][modality] = [
            data_g[modality]['x'][feature_name][:len_min]
            for feature_name in output_dict['feature_name_list'][modality]
        ]
        data_f_dict[filename][modality] = np.array(
            data_f_dict[filename][modality]).transpose()
        data_g_dict[filename][modality] = np.array(
            data_g_dict[filename][modality]).transpose()
    return data_f_dict, data_g_dict, output_dict


def concat_batch_list(lst):
    '''
    Function to join a list of datapoints.
    '''
    def concat_va(lst, va_str):
        # need to make sure that the last 20 frames are properly included
        va_out = []
        for l in range(len(lst)):
            if l + 1 == len(lst):
                va_out.append(lst[l][va_str])
            else:
                va_out.append(lst[l][va_str][:lst[l]['usr_update'].shape[0]])
        return np.concatenate(va_out)

    def concat_audio(lst, audio_str):
        # return torch.cat([l[audio_str] for l in lst])
        return np.concatenate([l[audio_str] for l in lst])

    def concat_y_strt_and_end(lst, y_str):
        # use for y_strt_f and y_end_f
        update_lengths = [l['usr_update'].shape[0] for l in lst]
        add_val = 0
        y_strt_out = []
        for i in range(len(lst)):
            y_strt_out.append(lst[i][y_str] + add_val)
            add_val += update_lengths[i]
        return y_strt_out

    def concat_simple(lst, simp_str):
        return [l[simp_str] for l in lst]

    def concat_sequence(lst, simp_str):
        return np.concatenate([l[simp_str] for l in lst])

    out = {
        'y_strt_f': concat_y_strt_and_end(lst, 'y_strt_f'),
        'y_strt_t': -1,
        'y_end_f':  concat_y_strt_and_end(lst, 'y_end_f'),
        'y_end_t': -1,
        'y_length': concat_simple(lst, 'y_length'),
        'associated_usr_ipu_strt_f': concat_y_strt_and_end(lst, 'associated_usr_ipu_strt_f'),
        'associated_usr_ipu_end_f': concat_y_strt_and_end(lst, 'associated_usr_ipu_end_f'),
        'usr_update': concat_audio(lst, 'usr_update'),
        'sys_update': concat_audio(lst, 'sys_update'),
        'sys_trn': concat_simple(lst, 'sys_trn'),
        'test_seq': [-1],
        'file': concat_simple(lst, 'file'),
        'a_usr': concat_simple(lst, 'a_usr'),
        'update_strt_t': concat_simple(lst, 'update_strt_t'),
        'update_end_t': concat_simple(lst, 'update_end_t'),
        'update_strt_f': concat_simple(lst, 'update_strt_f'),
        'update_end_f': concat_simple(lst, 'update_end_f'),
        'associated_usr_turn': concat_simple(lst, 'associated_usr_turn'),
        'update_idx': concat_simple(lst, 'update_idx'),
        'y_UT': concat_sequence(lst, 'y_UT'),
        'va_usr': concat_va(lst, 'va_usr'),
        'va_sys': concat_va(lst, 'va_sys'),
        'word_da_dict': concat_simple(lst, 'word_da_dict')
    }
    return out


def get_update_annots(ls_in):

    f_idx, file, a_usr, b_sys, annots_usr, annots_sys, data_usr, data_sys, num_feat_per_person, pad_noise_bool, n_pre = ls_in

    data_pts = []
    file_length = 0
    # framelen = 0.05
    framelen = 5
    pred_len_max = 20
    # turn_batch = 4

    # get covariance matrix
    backup_cov = np.array(json.load(open('./tools/mean_cov.json', 'rb')))
    # silence bools are for when booth speakers are silent
    silence_bools = np.where(annots_usr['prev_gap_silence_bools'])[0][1:]
    ipu_strts, ipu_ends = convert_to_ms_int_floor(
        annots_usr['ipu_start_times']), convert_to_ms_int_floor(annots_usr['ipu_end_times'])
    sil_strts, sil_ends = ipu_ends[silence_bools - 1], ipu_strts[silence_bools]

    try:
        na = np.where(sil_ends > sil_strts)[0]
        sil_strts, sil_ends = sil_strts[na], sil_ends[na]
        na = np.where(sil_strts[1:] >= sil_ends[:-1])[0]
        sil_strts[1:], sil_ends[1:] = sil_strts[na + 1], sil_ends[na + 1]
    except:
        print('error at file: ' + file)

    # assert all(sil_ends > sil_strts)
    # assert all(sil_strts[1:] > sil_ends[:-1])

    # use this to estimate the covariance matrix of the OPPOSITE person
    # we do this because there is less of a chance that the other person will be making
    # noises such as lip smacks etc.. (prob better reasons)
    ls = [np.arange(s, e)
          for s, e in zip(np.rint(sil_strts / framelen), np.rint(sil_ends / framelen)) if e > s]
    if len(ls):
        l = np.concatenate(ls)
        silences = data_sys[l.astype(np.int)]
    else:
        print('bad file')
        print(file)
        # pad with zeros  instead:
        silences = np.zeros([3, data_sys.shape[1]])

    # old covariance estimation
    # self.sil_cov_matrices[file][b_sys] = np.cov(silences, rowvar=False)
    # self.sil_means[file][b_sys] = np.mean(silences, 0)

    # if padding test_seqs with noise, estimate elliptic covariance matrix to avoid outliers
    sil_means = np.mean(silences, 0)

    if pad_noise_bool:
        try:
            cov = EllipticEnvelope().fit(silences - sil_means)
            cov = cov.covariance_
        except:
            cov = backup_cov
    else:
        cov = []
    # get va annotations
    max_time = int(np.rint(max([convert_to_ms_int_floor(annots_usr['end_time_words'][-1]),
                                convert_to_ms_int_floor(
                                    annots_sys['end_time_words'][-1])
                                ])/framelen))

    def get_va_annots(annots, max_time):
        va_annots = np.zeros(max_time, dtype=np.int16)
        for wrd_strt, wrd_end in zip(convert_to_ms_int_floor(annots['start_time_words']), convert_to_ms_int_floor(annots['end_time_words'])):
            wrd_strt_f = int(np.rint(wrd_strt / framelen))
            wrd_end_f = int(np.rint(wrd_end / framelen))
            # (maybe) need to add plus 1 because of floor operator
            va_annots[wrd_strt_f:wrd_end_f] = 1
        return va_annots

    va_annots_usr = get_va_annots(annots_usr, max_time)
    va_annots_sys = get_va_annots(annots_sys, max_time)

    # pad with extra values for predictions
    va_annots_usr = np.concatenate(
        [va_annots_usr, np.zeros(pred_len_max+1, dtype=np.int16)])
    va_annots_sys = np.concatenate(
        [va_annots_sys, np.zeros(pred_len_max+1, dtype=np.int16)])
    # hs_annots_sys = get_hs_annots(annots_sys, va_annots_usr, max_time)

    sys_update_start_frames = np.rint(convert_to_ms_int_floor(
        annots_usr['updates']['sys_update_strt_times'])[:-1] / framelen).astype(np.int32)
    sys_update_end_frames = np.rint(convert_to_ms_int_floor(
        annots_usr['updates']['sys_update_end_times'])[:-1] / framelen).astype(np.int32)

    # num_updates = len(annots_usr['updates']['sys_update_turns']) - 1 # ommit last update
    usr_updates, sys_updates, sys_turns = [], [], []
    update_batch_list = []
    for update_idx, (strt_fidx_update, end_fidx_update) in enumerate(zip(sys_update_start_frames, sys_update_end_frames)):  # update_idx
        strt_t_update = convert_to_ms_int_floor(
            annots_usr['updates']['sys_update_strt_times'][update_idx])
        end_t_update = convert_to_ms_int_floor(
            annots_usr['updates']['sys_update_end_times'][update_idx])

        if strt_fidx_update == end_fidx_update:
            print('Update is zero length')
            pdb.set_trace()
            strt_fidx_update = strt_fidx_update - 1

        # Get associated turns for the user
        usr_turn_words_start_time_ms_int = convert_to_ms_int_floor(
            annots_usr['turn_words_start_time'])
        usr_turn_words_end_time_ms_int = convert_to_ms_int_floor(
            annots_usr['turn_words_end_time'])
        usr_update_turns = \
            (usr_turn_words_start_time_ms_int < strt_t_update) & (usr_turn_words_end_time_ms_int >= strt_t_update) | \
            (usr_turn_words_start_time_ms_int >= strt_t_update) & (usr_turn_words_start_time_ms_int < end_t_update) | \
            (usr_turn_words_start_time_ms_int < end_t_update) & (
                usr_turn_words_end_time_ms_int >= end_t_update)
        usr_update_turns = np.where(usr_update_turns)[0]
        usr_update_turn_starts_t = annots_usr['turn_words_start_time'][usr_update_turns]
        usr_update_turn_ends_t = annots_usr['turn_words_end_time'][usr_update_turns]
        sys_update_turn_start_t = annots_sys['turn_words_start_time'][update_idx]
        sys_update_turn_end_t = annots_sys['turn_words_end_time'][update_idx]
        sys_turn_enc_strt_f = int(
            np.rint(convert_to_ms_int_floor(sys_update_turn_start_t) / framelen))
        sys_turn_enc_end_f = int(
            np.rint(convert_to_ms_int_floor(sys_update_turn_end_t) / framelen))
        assert convert_to_ms_int_floor(sys_update_turn_end_t) == end_t_update
        sys_turn_full_over = annots_sys['turn_full_overlap'][update_idx]
        # If system turn is not in full overlap, get the user turn that it is associated with and the offset

        if not (sys_turn_full_over or update_idx == 0):
            # Associated user turn is the user turn that began directly before the system turn
            associated_usr_turn = usr_update_turns[np.where(convert_to_ms_int_floor(
                usr_update_turn_starts_t) < convert_to_ms_int_floor(sys_update_turn_start_t))[0][-1]]
            # hack. Also, catch other overlaps that aren't in sys_turn_full_over
            if annots_usr['turn_words_end_time'][associated_usr_turn] > sys_update_turn_end_t:
                associated_usr_turn = -1
        else:
            associated_usr_turn = -1

        # Get associated IPUs for the user
        usr_ipu_start_time_ms_int = convert_to_ms_int_floor(
            annots_usr['ipu_start_times'])
        usr_ipu_end_time_ms_int = convert_to_ms_int_floor(
            annots_usr['ipu_end_times'])
        usr_update_ipus = \
            (usr_ipu_start_time_ms_int < strt_t_update) & (usr_ipu_end_time_ms_int >= strt_t_update) | \
            (usr_ipu_start_time_ms_int >= strt_t_update) & (usr_ipu_start_time_ms_int < end_t_update) | \
            (usr_ipu_start_time_ms_int < end_t_update) & (
                usr_ipu_end_time_ms_int >= end_t_update)
        usr_update_ipus = np.where(usr_update_ipus)[0]
        usr_update_ipus_starts_t = annots_usr['ipu_start_times'][usr_update_ipus]
        usr_update_ipus_ends_t = annots_usr['ipu_end_times'][usr_update_ipus]

        if update_idx == 0:
            associated_usr_ipu = -1
            associated_usr_ipu_strt_t = 0
            associated_usr_ipu_end_t = -1
            associated_usr_ipu_strt_f = 0
            associated_usr_ipu_end_f = -1
        # If system turn is not in full overlap, get the user turn that it is associated with and the offset
        elif not (associated_usr_turn == -1):
            # Associated user IPU is the user ipu that began directly before the system turn
            associated_usr_ipu = usr_update_ipus[np.where(convert_to_ms_int_floor(
                usr_update_ipus_starts_t) < convert_to_ms_int_floor(sys_update_turn_start_t))[0][-1]]
            associated_usr_ipu_strt_t = annots_usr['ipu_start_times'][associated_usr_ipu]
            associated_usr_ipu_end_t = annots_usr['ipu_end_times'][associated_usr_ipu]
            associated_usr_ipu_strt_f = int(
                np.rint(convert_to_ms_int_floor(associated_usr_ipu_strt_t) / framelen))
            associated_usr_ipu_end_f = int(
                np.rint(convert_to_ms_int_floor(associated_usr_ipu_end_t) / framelen))

        else:
            associated_usr_ipu = np.where(convert_to_ms_int_floor(
                annots_usr['ipu_start_times']) < convert_to_ms_int_floor(sys_update_turn_start_t))[0][-1]
            associated_usr_ipu_strt_t = annots_usr['ipu_start_times'][associated_usr_ipu]
            associated_usr_ipu_end_t = annots_usr['ipu_end_times'][associated_usr_ipu]
            associated_usr_ipu_strt_f = int(
                np.rint(convert_to_ms_int_floor(associated_usr_ipu_strt_t) / framelen))
            associated_usr_ipu_end_f = int(
                np.rint(convert_to_ms_int_floor(associated_usr_ipu_end_t) / framelen))
            associated_usr_ipu = -1

        # get updates
        usr_update = data_usr[strt_fidx_update:end_fidx_update]
        sys_update = data_sys[strt_fidx_update:end_fidx_update]

        # continuous voice activity annotations
        cont_pred_vec_usr = va_annots_usr[strt_fidx_update:end_fidx_update + 20]
        cont_pred_vec_sys = va_annots_sys[strt_fidx_update:end_fidx_update + 20]

        # Get system turn for encoder
        sys_enc_feats = data_sys[sys_turn_enc_strt_f: sys_turn_enc_end_f]

        # Get test_seq
        # Find the first switch from silence to speech by the user after the system ground truth start and pad with silence noise.
        sil_indx = 0
        while sil_indx < len(va_annots_usr[sys_turn_enc_strt_f:])-1:
            if va_annots_usr[sys_turn_enc_strt_f:][sil_indx] == 0 and va_annots_usr[sys_turn_enc_strt_f:][sil_indx + 1] == 1:
                break
            else:
                sil_indx += 1

        # sil indx is one frame before the last of the silence frames
        sil_indx = sys_turn_enc_strt_f + sil_indx
        if (sil_indx - strt_fidx_update) == 0:
            sil_indx += 1

        test_seq = data_usr[strt_fidx_update:sil_indx]

        try:
            assert test_seq.shape[0] > 0
        except AssertionError:
            print('test seq shape is zero in file: ' + file)

        # Get train Y
        y_UT = np.zeros(len(usr_update), dtype=np.int16)
        # protect against turns that start on first frame of file
        y_train_strt = max([0, sys_turn_enc_strt_f - 1 - strt_fidx_update])
        y_UT[y_train_strt: sys_turn_enc_end_f - strt_fidx_update - 1] = 1

        if not any(y_UT == 1):
            print('bad')
        y_strt_t = sys_update_turn_start_t - \
            annots_usr['updates']['sys_update_strt_times'][update_idx]
        y_end_t = sys_update_turn_end_t - \
            annots_usr['updates']['sys_update_strt_times'][update_idx]
        y_strt_f = sys_turn_enc_strt_f - strt_fidx_update
        y_end_f = sys_turn_enc_end_f - strt_fidx_update
        associated_usr_ipu_strt_f = associated_usr_ipu_strt_f - strt_fidx_update
        associated_usr_ipu_end_f = associated_usr_ipu_end_f - strt_fidx_update

        # Get words
        # Candidate system turn encoding words and update words
        s_i = annots_sys['turn_words_start_indx'][update_idx]
        e_i = annots_sys['turn_words_end_indx'][update_idx] + 1
        sys_enc_words = annots_sys['target_words'][s_i:e_i]
        sys_enc_word_strt_ts = annots_sys['start_time_words'][s_i:e_i]
        sys_enc_word_end_ts = annots_sys['end_time_words'][s_i:e_i]
        sys_update_word_strt_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_word_strt_ts) / framelen) - strt_fidx_update
        sys_update_word_end_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_word_end_ts) / framelen) - strt_fidx_update
        sys_enc_word_strt_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_word_strt_ts) / framelen) - sys_turn_enc_strt_f
        sys_enc_word_end_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_word_end_ts) / framelen) - sys_turn_enc_strt_f

        # User update words
        if not len(usr_update_turns):
            s_i, s_e = 0, 0
        else:
            s_i = annots_usr['turn_words_start_indx'][usr_update_turns][0]
            e_i = annots_usr['turn_words_end_indx'][usr_update_turns][-1] + 1
        usr_update_words = annots_usr['target_words'][s_i:e_i]
        usr_update_word_strt_ts = annots_usr['start_time_words'][s_i:e_i]
        usr_update_word_end_ts = annots_usr['end_time_words'][s_i:e_i]
        usr_update_word_strt_frames = np.rint(convert_to_ms_int_floor(
            usr_update_word_strt_ts) / framelen) - strt_fidx_update
        usr_update_word_end_frames = np.rint(convert_to_ms_int_floor(
            usr_update_word_end_ts) / framelen) - strt_fidx_update

        # test seq words
        usr_end_fs = np.rint(convert_to_ms_int_floor(
            annots_usr['end_time_words']) / framelen)
        test_wrd_indices = np.where(
            (usr_end_fs >= strt_fidx_update) & (usr_end_fs < sil_indx))[0]
        if not len(test_wrd_indices):
            s_i, e_i = 0, 0
        else:
            s_i, e_i = test_wrd_indices[0], test_wrd_indices[-1]+1
        test_words = annots_usr['target_words'][s_i:e_i]
        test_word_strt_ts = annots_usr['start_time_words'][s_i:e_i]
        test_word_end_ts = annots_usr['end_time_words'][s_i:e_i]
        test_word_strt_frames = np.rint(convert_to_ms_int_floor(
            test_word_strt_ts) / framelen) - strt_fidx_update
        test_word_end_frames = np.rint(convert_to_ms_int_floor(
            test_word_end_ts) / framelen) - strt_fidx_update

        # dialogue acts for sys encoding
        turn_ipu_start_indx = annots_sys['turn_ipu_start_indx'][update_idx]
        turn_ipu_end_indx = annots_sys['turn_ipu_start_indx'][update_idx+1]
        # sys_enc_das = annots_sys['da_ISO_second_pass_vec'][turn_ipu_start_indx:turn_ipu_end_indx]
        sys_enc_da_strt_ts = annots_sys['ipu_start_times'][turn_ipu_start_indx:turn_ipu_end_indx]
        sys_enc_da_end_ts = annots_sys['ipu_end_times'][turn_ipu_start_indx:turn_ipu_end_indx]
        sys_enc_da_strt_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_da_strt_ts) / framelen) - sys_turn_enc_strt_f
        sys_enc_da_end_frames = np.rint(convert_to_ms_int_floor(
            sys_enc_da_end_ts) / framelen) - sys_turn_enc_strt_f

        word_da_dict = {
            'strt_t_update': strt_t_update,
            'end_t_update': end_t_update,
            'strt_fidx_update': strt_fidx_update,
            'end_fidx_update': end_fidx_update,
            'sys_enc_words': sys_enc_words,
            'sys_enc_word_strt_ts': sys_enc_word_strt_ts,
            'sys_enc_word_end_ts': sys_enc_word_end_ts,
            'sys_update_words': sys_enc_words,
            'sys_update_word_strt_frames': sys_update_word_strt_frames.astype(np.int16),
            'sys_update_word_end_frames': sys_update_word_end_frames.astype(np.int16),
            'sys_enc_word_strt_frames': sys_enc_word_strt_frames.astype(np.int16),
            'sys_enc_word_end_frames': sys_enc_word_end_frames.astype(np.int16),
            'usr_update_words': usr_update_words,
            'usr_update_word_strt_ts': usr_update_word_strt_ts,
            'usr_update_word_end_ts': usr_update_word_end_ts,
            'usr_update_word_strt_frames': usr_update_word_strt_frames.astype(np.int16),
            'usr_update_word_end_frames': usr_update_word_end_frames.astype(np.int16),
            'test_words': test_words,
            'test_word_strt_ts': test_word_strt_ts,
            'test_word_end_ts': test_word_end_ts,
            'test_word_strt_frames': test_word_strt_frames.astype(np.int16),
            'test_word_end_frames': test_word_end_frames.astype(np.int16),
            # 'sys_enc_das': sys_enc_das,
            'sys_enc_da_strt_ts': sys_enc_da_strt_ts,
            'sys_enc_da_end_ts': sys_enc_da_end_ts,
            'sys_enc_da_strt_frames': sys_enc_da_strt_frames,
            'sys_enc_da_end_frames': sys_enc_da_end_frames
        }

        data_pts.append(
            {
                'y_strt_f': [y_strt_f],
                'y_strt_t': [y_strt_t],
                'y_end_f': [y_end_f],
                'y_end_t': [y_end_t],
                'y_length': [len(sys_enc_feats)],
                'associated_usr_ipu_strt_f': [associated_usr_ipu_strt_f],
                'associated_usr_ipu_end_f': [associated_usr_ipu_end_f],
                'usr_update': usr_update,
                'sys_update': sys_update,
                'sys_trn': [sys_enc_feats],
                'test_seq': test_seq,
                'file': [file],
                'a_usr': [a_usr],
                'update_strt_t': [annots_usr['updates']['sys_update_strt_times'][update_idx]],
                'update_end_t': [annots_usr['updates']['sys_update_end_times'][update_idx]],
                'update_strt_f': [strt_fidx_update],
                'update_end_f': [end_fidx_update],
                'associated_usr_turn': associated_usr_turn,
                'update_idx': update_idx,
                'y_UT': y_UT,
                'va_usr': cont_pred_vec_usr,
                'va_sys': cont_pred_vec_sys,
                'word_da_dict': word_da_dict
            }
        )
        file_length += 1
    return [data_pts, sil_means, cov, file, a_usr, b_sys, file_length]


def get_fold_name(naming_dict, lstm_sets_dict):
    name_str = ''
    name_str += './results/'
    name_str += naming_dict['time_str']
    name_str += '_batsz_' + str(naming_dict['batch_size'])
    # name_str += '_padmaxlen_' + str(naming_dict['lstm_sets_dict']['pad_all_max_len_bool'])
    name_str += '_padframes_' + \
        str(naming_dict['lstm_sets_dict']['extra_pad_frames'])
    name_str += '_l2_' + str(lstm_sets_dict['l2'])[2:]
    name_str += '_lr_' + str(lstm_sets_dict['learning_rate'])
    name_str += '_turnBtch2_' + str(lstm_sets_dict['two_sys_turn'])
    # name_str += '_sysEnc_' + str(naming_dict['lstm_sets_dict']['response_encoder_hidden_size'])
    # name_str += '_mastEnc_' + str(naming_dict['lstm_sets_dict']['master_encoder_hidden_size'])
    # name_str += '_inf_' + str(naming_dict['lstm_sets_dict']['inference_hidden_size'])
    name_str += '_skipVAE_' + \
        str(naming_dict['lstm_sets_dict']['encoder_settings']['skip_vae'])
    name_str += '_encAbl_' + \
        str(naming_dict['lstm_sets_dict']['enc_ablation_setting'])
    name_str += '_decAbl_' + \
        str(naming_dict['lstm_sets_dict']['dec_ablation_setting'])
    name_str += '_maxStrtTest_' + \
        str(naming_dict['lstm_sets_dict']['max_strt_test'])
    name_str += '_maxWaitTrain_' + \
        str(naming_dict['lstm_sets_dict']['max_wait_train'])
    name_str += '_KL_' + \
        str(naming_dict['lstm_sets_dict']['pred_task_dict']['KLD']['weight'])
    # name_str += '_TL_' + str(naming_dict['lstm_sets_dict']['pred_task_dict']['TL']['weight'])
    name_str += '_fullTest_' + \
        str(naming_dict['lstm_sets_dict']['full_test_flag'])
    name_str += '_seed_' + str(naming_dict['lstm_sets_dict']['seed'])
    # name_str += '_useling_' + str(lstm_sets_dict['use_ling'])
    # name_str += '_freezeLing_' + str(lstm_sets_dict['ling_emb_freeze'])
    # name_str += '_useAcous_' + str(lstm_sets_dict['encoder_settings']['use_acous'])
    # name_str += '_encUseLing_' + str(lstm_sets_dict['encoder_settings']['use_ling'])
    name_str += '_valid_test' if lstm_sets_dict['test_valid'] and lstm_sets_dict['just_test'] else ''
    name_str += '_vae_exp_'+lstm_sets_dict['vae_target_da']+'_'+str(
        lstm_sets_dict['vae_data_multiplier']) if lstm_sets_dict['vae_experiments'] else ''

    name_str += '~'+str(naming_dict['note'])

    naming_dict['fold_name'] = name_str
    return naming_dict
