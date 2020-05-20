import json
import os
import pdb
import pickle
import time as t
from collections import OrderedDict

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

# from util import get_cont_hold_shift, get_hold_shift
import settings
from settings import (batch_size, framelen, lstm_sets_dict, naming_dict,
                      out_str, pred_task_dict)

loss_func_CE = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
loss_func_BCE_logit = torch.nn.BCEWithLogitsLoss(reduction='sum')
loss_func_BCE = torch.nn.BCELoss(reduction='sum')
# loss_func_MSE = torch.nn.MSELoss(reduction='sum')
loss_func_MSE = torch.nn.L1Loss(reduction='sum')
tanh_func = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()


def sanity_check_func(data, rand_strt_dict, input_dataloader):

    vars = ['ud_strt', 'minRndStrt', 'maxRndStrt', 'sampRndStrt', 'trgtNT',
            'samp', 'out_probs', 'va_usr', 'loud_usr', 'va_sys', 'loud_sys']

    var_dict = {var: [] for var in vars}
    loudness_indx = 46
    for dat_idx, dat in enumerate(data):
        file = lstm_sets_dict['sanity_check_file'][0]
        probs_dict, y_trgt_stats_dict, samp_out, batch_dict, sys_enc_out_save, sys_enc_mu_save, sys_enc_log_var_save, rand_strt_dict_save = dat
        dataset = input_dataloader.dataset
        files = [dataset.file_list[f] for f in batch_dict['file_idx']]
        a_usrs = [['A', 'B'][a] for a in batch_dict['a_idx']]
        sel_array = np.array([1 if f == file and a_b == 'A' else 0 for f, a_b in zip(
            files, a_usrs)]).astype(np.bool)
        dset_idx = batch_dict['dataset_idx'][sel_array].squeeze()
        if any(sel_array):
            sel_array = np.where(sel_array)[0]
            sys_update = dataset.dataset[dset_idx]['sys_update']
            update_length = sys_update.shape[0]
            var_dict['loud_sys'].append(
                sys_update[:, loudness_indx] - min(sys_update[:, loudness_indx]))
            usr_update = dataset.dataset[dset_idx]['usr_update']
            var_dict['loud_usr'].append(
                usr_update[:, loudness_indx] - min(usr_update[:, loudness_indx]))
            probs = probs_dict['NT'][sel_array].squeeze()[:update_length]
            probs[probs < 0.0] = probs[probs < 0.0]*0.1
            var_dict['out_probs'].append(probs[:update_length])
            var_dict['va_usr'].append(
                dataset.dataset[dset_idx]['va_usr'][:update_length])
            var_dict['va_sys'].append(
                dataset.dataset[dset_idx]['va_sys'][:update_length])
            ass_usr_ipu_strt_f = int(
                batch_dict['associated_usr_ipu_strt_f'][sel_array])
            var_dict['samp'].append(
                samp_out[sel_array, :update_length].squeeze())
            update_strt = np.zeros(update_length).astype(np.int16)
            update_strt[0] = 1
            var_dict['ud_strt'].append(update_strt)
            var_dict['trgtNT'].append(
                y_trgt_stats_dict['NT'][sel_array].squeeze()[:update_length])
            # var_dict['trgtTL'].append(y_trgt_stats_dict['TL'][sel_array].squeeze()[:update_length])
            min_rnd_strt = np.zeros(update_length).astype(np.int16)
            max_rnd_strt = np.zeros(update_length).astype(np.int16)
            samp_rnd_strt = np.zeros(update_length).astype(np.int16)
            min_rnd_strt[rand_strt_dict[dat_idx]
                         ['min_rand_strt_f'][sel_array]] = 1
            max_rnd_strt[rand_strt_dict[dat_idx]
                         ['max_rand_strt_f'][sel_array]] = 1
            samp_rnd_strt[rand_strt_dict[dat_idx]
                          ['sampled_rand_strt'][sel_array]] = 1
            var_dict['minRndStrt'].append(min_rnd_strt)
            var_dict['maxRndStrt'].append(max_rnd_strt)
            var_dict['sampRndStrt'].append(samp_rnd_strt)
    var_dict_2 = {}
    for k, i in var_dict.items():
        i_2 = []
        for it in i:
            if not (len(it.shape) == 1):
                i_2.append(np.squeeze(it))
            else:
                i_2.append(it)

        var_dict_2[k] = np.concatenate(i_2)

    def get_plot_points(strts, ends, heights):
        xs, ys, = [], []
        for strt, nd, height in zip(strts, ends, heights):
            xs += [strt, strt, nd, nd]
            ys += [0, height, height, 0]
        return xs, ys
    plt.rcParams.update({'font.size': 7})
    fig, axes = plt.subplots(len(vars), 1, sharex=True)

    for var, ax in zip(vars, axes):
        plot_points = np.array(list(range(len(var_dict_2[var]))))*0.05
        p_x, p_y = get_plot_points(
            plot_points[:-1], plot_points[1:], var_dict_2[var][:-1])
        # p_x = list(range(len(plot_vals)))
        # p_y = plot_vals
        ax.fill_between(p_x, 0, p_y)
        ax.set_ylabel(var)
        # plt.savefig('./myplt.eps')

    # axes[-2].plot(0.05*np.arange(len(data_usr[:, 46])), data_usr[:, 46])
    # axes[-2].set_ylabel('usr_feat')
    # axes[-1].plot(0.05*np.arange(len(data_usr[:, 46])), data_sys[:, 46])
    # axes[-1].set_ylabel('sys_feat')
    title_str = files[0]+'_'+a_usrs[0]
    plt.suptitle(title_str)
    plt.savefig('./myplt.eps')
    pickle.dump(fig, open('./sanity_check_plots/FigureObject.fig.pickle', 'wb'))
    print('')


def get_test_stats(update_annots_test, stats_data, test_dataloader, test_files_list, epoch):

    offset_data_types = ['true_offset', 'pred_offset', 'timed_out', 'sys_full_overlap', 'sys_one_da',
                         'usr_end_da_swbd', 'usr_end_da_nite', 'sys_strt_da_swbd', 'sys_strt_da_nite']
    offset_data_types += ['file', 'a_usr', 'usr_trn_strt_t', 'usr_trn_end_t',
                          'sys_trn_strt_t', 'sys_trn_end_t', 'usr_turn_idx', 'sys_trn_idx']
    offset_data_types += ['min_rand_strt_f',
                          'max_rand_strt_f', 'sampled_rand_strt']

    sys_enc_data_out, sys_enc_mu_data_out, sys_enc_log_var_data_out = [], [], []
    offset_data = {o: [] for o in offset_data_types}
    test_files = test_files_list
    # pred_len = test_dataloader.dataset.pred_task_dict['VA']['pred_len']
    time_out = test_dataloader.dataset.time_out_length
    preds_dict = {}
    active_outputs = test_dataloader.dataset.active_outputs
    for prd in active_outputs:
        preds_dict[prd] = {}
        for file in test_files:
            preds_dict[prd][file] = {}
            file_length = max([update_annots_test[file]['A']['end_time_words']
                               [-1], update_annots_test[file]['B']['end_time_words'][-1]])
            file_length = int(np.round(file_length/framelen))
            for a_b in ['A', 'B']:
                pred_len = test_dataloader.dataset.pred_task_dict[prd]['pred_len']
                preds_dict[prd][file][a_b] = np.squeeze(
                    np.zeros([file_length, pred_len], dtype=np.float32))

    print('getting stats data')
    for dat in stats_data:
        probs_dict, y_trgt_stats_dict, samp_out, batch, sys_enc_out_save, sys_enc_mu_save, sys_enc_log_var_save, rand_strt_dict = dat
        files = [test_dataloader.dataset.file_list[f]
                 for f in batch['file_idx']]
        a_usrs = [['A', 'B'][a] for a in batch['a_idx']]
        usr_update_idxs = batch['ass_usr_turn']
        sys_trn_idxs = batch['update_idx']
        sel_batch_indices = np.where(~(np.array(usr_update_idxs) == -1))[0]
        for sel_y_trgt, sel_bat in enumerate(sel_batch_indices):
            file, a_usr, usr_turn_idx, sys_trn_idx = files[sel_bat], a_usrs[
                sel_bat], usr_update_idxs[sel_bat], sys_trn_idxs[sel_bat]
            # update_strt_f, update_end_f = batch['update_strt_f'][sel_bat][0], batch['update_end_f'][sel_bat][0]
            update_strt_t, update_end_t = batch['update_strt_t'][sel_bat][0], batch['update_end_t'][sel_bat][0]
            # sys_enc_out_save_bat = sys_enc_out_save[sel_y_trgt]
            sys_enc_out_save_bat = sys_enc_out_save[sel_bat]
            sys_enc_mu_out = sys_enc_mu_save[sel_bat]
            sys_enc_log_var_out = sys_enc_log_var_save[sel_bat]
            min_rand_strt_f = int(rand_strt_dict['min_rand_strt_f'][sel_bat])
            max_rand_strt_f = int(rand_strt_dict['max_rand_strt_f'][sel_bat])
            sampled_rand_strt = int(
                rand_strt_dict['sampled_rand_strt'][sel_bat])
            ass_usr_ipu_strt_f = batch['associated_usr_ipu_strt_f'][sel_bat][0]
            b_sys = [a_b for a_b in ['A', 'B'] if not (a_usr == a_b)][0]

            # Collect necessary info
            usr_trn_strt_t = update_annots_test[file][a_usr]['turn_words_start_time'][usr_turn_idx]
            usr_trn_end_t = update_annots_test[file][a_usr]['turn_words_end_time'][usr_turn_idx]
            sys_trn_strt_t = update_annots_test[file][b_sys]['turn_words_start_time'][sys_trn_idx]
            sys_trn_end_t = update_annots_test[file][b_sys]['turn_words_end_time'][sys_trn_idx]

            sys_turn_end_da_nite = update_annots_test[file][b_sys]['turn_words_end_da_nite'][sys_trn_idx]
            sys_turn_strt_da_nite = update_annots_test[file][b_sys]['turn_words_start_da_nite'][sys_trn_idx]
            sys_turn_end_da_swbd = update_annots_test[file][b_sys]['turn_words_end_swbdType'][sys_trn_idx]
            sys_turn_strt_da_swbd = update_annots_test[file][b_sys]['turn_words_start_swbdType'][sys_trn_idx]
            usr_turn_end_da_nite = update_annots_test[file][a_usr]['turn_words_end_da_nite'][usr_turn_idx]
            usr_turn_strt_da_nite = update_annots_test[file][a_usr]['turn_words_start_da_nite'][usr_turn_idx]
            usr_turn_end_da_swbd = update_annots_test[file][a_usr]['turn_words_end_swbdType'][usr_turn_idx]
            usr_turn_strt_da_swbd = update_annots_test[file][a_usr]['turn_words_start_swbdType'][usr_turn_idx]

            # other sys details
            sys_turn_full_overlap = update_annots_test[file][b_sys]['turn_full_overlap'][sys_trn_idx]
            sys_turn_one_da_bool = update_annots_test[file][b_sys]['turn_one_da_bool'][sys_trn_idx]

            first_trig_idx = np.where(samp_out[sel_bat] == 1)[0]
            first_non_negative = np.where(samp_out[sel_bat] != -1)[0][0]
            last_possible = np.where(samp_out[sel_bat] == -2)[0]
            if not len(last_possible):
                last_possible = len(samp_out[sel_bat])
            else:
                last_possible = last_possible[0]

            # Calculate trigger index for samp_out
            y_trgt_offset_from_strt_usr_update_fr = int(
                np.where(y_trgt_stats_dict[out_str][sel_bat] == 1)[0][0] + 1)
            y_trgt_offset_from_strt_usr_update = framelen * \
                y_trgt_offset_from_strt_usr_update_fr
            if not (len(first_trig_idx)) or (first_trig_idx[0] > (y_trgt_offset_from_strt_usr_update_fr + time_out)):
                samp_sys_trig_offset_from_strt_usr_update = int(
                    y_trgt_offset_from_strt_usr_update_fr + time_out)
                timed_out = True
            else:
                # need plus one because user starts speaking
                samp_sys_trig_offset_from_strt_usr_update = first_trig_idx[0] + 1
                timed_out = False

            # Calculate offsets
            true_offset = sys_trn_strt_t - usr_trn_end_t
            usr_end_frame_from_strt_usr_update = (
                usr_trn_end_t // framelen - update_strt_t // framelen)  # from annotations
            usr_end_time_from_strt_usr_update = usr_trn_end_t - update_strt_t

            # quantized true_offset
            # good (quantized) # this should be the best
            pred_offset = framelen * \
                (samp_sys_trig_offset_from_strt_usr_update -
                 usr_end_frame_from_strt_usr_update)

            # add the data
            offset_data['true_offset'].append(true_offset)
            offset_data['pred_offset'].append(pred_offset)
            offset_data['timed_out'].append(timed_out)
            offset_data['usr_end_da_swbd'].append(usr_turn_end_da_swbd)
            offset_data['usr_end_da_nite'].append(usr_turn_end_da_nite)
            offset_data['sys_strt_da_swbd'].append(sys_turn_strt_da_swbd)
            offset_data['sys_strt_da_nite'].append(sys_turn_strt_da_nite)
            offset_data['sys_full_overlap'].append(int(sys_turn_full_overlap))
            offset_data['sys_one_da'].append(int(sys_turn_one_da_bool))
            offset_data['file'].append(file)
            offset_data['a_usr'].append(a_usr)
            offset_data['usr_trn_strt_t'].append(float(usr_trn_strt_t))
            offset_data['usr_trn_end_t'].append(float(usr_trn_end_t))
            offset_data['sys_trn_strt_t'].append(float(sys_trn_strt_t))
            offset_data['sys_trn_end_t'].append(float(sys_trn_end_t))
            offset_data['usr_turn_idx'].append(int(usr_turn_idx))
            offset_data['sys_trn_idx'].append(int(sys_trn_idx))

            sys_enc_data_out.append(sys_enc_out_save_bat)
            sys_enc_mu_data_out.append(sys_enc_mu_out)
            sys_enc_log_var_data_out.append(sys_enc_log_var_out)

            offset_data['min_rand_strt_f'].append(min_rand_strt_f)
            offset_data['max_rand_strt_f'].append(max_rand_strt_f)
            offset_data['sampled_rand_strt'].append(sampled_rand_strt)

    # Plots
    plots_root_path = naming_dict['fold_name']+'/plots/'
    plots_da_path = plots_root_path + 'DAs_'+str(epoch)+'/'
    if not os.path.exists(plots_root_path):
        os.mkdir(plots_root_path)
    if not os.path.exists(plots_da_path):
        os.mkdir(plots_da_path)

    # for da in da_type_list:
    def da_stats(true_offset, pred_offset):
        da_dict = OrderedDict()
        true_offset, pred_offset = np.array(true_offset), np.array(pred_offset)

        # mean, variance,MAE, MSE, KL, JS
        da_dict['mean'] = float(np.mean(pred_offset))
        da_dict['variance'] = float(np.var(pred_offset))
        da_dict['median'] = float(np.median(pred_offset))
        da_dict['mae'] = float(
            np.sum(np.abs(true_offset - pred_offset)) / len(true_offset))
        da_dict['mse'] = float(
            np.sum(np.square(true_offset - pred_offset)) / len(true_offset))
        true_offset_hist = scipy.histogram(
            true_offset, bins=60, range=(-1.5, 1.5))[0]
        pred_offset_hist = scipy.histogram(
            pred_offset, bins=60, range=(-1.5, 1.5))[0]
        da_dict['forward_ent'] = float(
            scipy.stats.entropy(true_offset_hist, pred_offset_hist))
        da_dict['backwards_ent'] = float(
            scipy.stats.entropy(pred_offset_hist, true_offset_hist))
        # M = 0.5 * (pred_offset_hist + true_offset_hist)
        # da_dict['jensen_shannon'] = float(scipy.spatial.distance.jensenshannon(pred_offset_hist / np.sum(pred_offset_hist),
        #  true_offset_hist/sum(true_offset_hist)))
        # da_dict['jensen_shannon'] = float(np.sqrt((da_dict['forward_ent']+da_dict['backwards_ent'])/2))
        return da_dict

    def hist_plot(points_list, name_list, title, path):
        # plt.figure()
        for points in points_list:
            plt.hist(points, bins=np.linspace(-1.5, 1.5, 60),
                     histtype=u'step', density=True)
        plt.legend(name_list)
        plt.title(title)
        plt.xlabel('Offset (Seconds)')
        plt.savefig(path)
        plt.cla()

    stats_object = OrderedDict()
    # Get combined stats
    stats_object['combined'] = da_stats(
        offset_data['true_offset'], offset_data['pred_offset'])
    points_list = [offset_data['true_offset'], offset_data['pred_offset']]
    name_list = ['True', 'Predicted']
    hist_plot(points_list, name_list, 'Distribution of System Turn Offsets',
              plots_da_path+'combined.pdf')

    # Get DA stats and plots
    # choose the five most common DAs
    stats_object['DAs'] = OrderedDict()
    # da_type_list = list(set(offset_data['usr_end_da_nite']).intersection(set(offset_data['sys_strt_da_nite'])))
    da_counter = {}
    for da in offset_data['sys_strt_da_nite']:
        if da in da_counter:
            da_counter[da] += 1
        else:
            da_counter[da] = 1

    da_type_list = ['statement', 'backchannel', 'opinion',
                    'agree', 'apprec', 'yn_q', 'yes', 'no', 'wh_q']
    for da_type in da_type_list:

        da_indx = np.where(da_type == np.array(
            offset_data['sys_strt_da_nite']))[0]
        true_off = np.array(offset_data['true_offset'])[da_indx]
        pred_off = np.array(offset_data['pred_offset'])[da_indx]
        points_list = [true_off, pred_off, offset_data['true_offset']]
        name_list = ['True', 'Predicted', 'All']
        title = 'Distribution of '+da_type + ' offsets'
        path = plots_da_path + title + '.pdf'
        stats_object['DAs'][da_type] = da_stats(true_off, pred_off)
        hist_plot(points_list, name_list, title, path)

    # Plot no/yes on same plot
    da_indx_yes = np.where('yes' == np.array(
        offset_data['sys_strt_da_nite']))[0]
    da_indx_no = np.where('no' == np.array(offset_data['sys_strt_da_nite']))[0]
    points_list = [np.array(offset_data['true_offset'])[da_indx_yes], np.array(offset_data['true_offset'])[da_indx_no],
                   np.array(offset_data['pred_offset'])[da_indx_yes], np.array(offset_data['pred_offset'])[da_indx_no]]
    name_list = ['True "Yes"', 'True "No"',
                 'Predicted "Yes"', 'Predicted "No"']
    title = 'Distribution of "Yes" and "No" offsets'
    path = plots_da_path + title + '.pdf'
    hist_plot(points_list, name_list, title, path)
    # save offset data
    json.dump(offset_data, open(
        naming_dict['fold_name'] + '/offset_data.json', 'w'), indent=4)
    json.dump(stats_object, open(
        naming_dict['fold_name'] + '/stats_object.json', 'w'), indent=4)
    sys_enc_data_out = np.array(sys_enc_data_out)
    pickle.dump(sys_enc_data_out, open(
        naming_dict['fold_name'] + '/sys_encodings.p', 'wb'))
    sys_enc_log_var_data_out = np.array(sys_enc_log_var_data_out)
    pickle.dump(sys_enc_log_var_data_out, open(
        naming_dict['fold_name'] + '/sys_encode_log_var.p', 'wb'))
    sys_enc_mu_data_out = np.array(sys_enc_mu_data_out)
    pickle.dump(sys_enc_mu_data_out, open(
        naming_dict['fold_name'] + '/sys_encode_mu.p', 'wb'))

    # return out_hold_shift, out_cont_hold_shift, stats_object
    return [], [], stats_object


def get_batch_items_for_full_test(batch):

    batch_keys = ['y_strt_f', 'y_strt_t', 'y_end_f', 'y_length', 'associated_usr_ipu_strt_f',
                  'update_strt_f', 'update_end_f',
                  'update_strt_t', 'update_end_t', 'update_idx', 'ass_usr_turn',
                  'file_idx', 'a_idx', 'dataset_idx']
    batch_turn_info = {k: i.data.cpu().numpy()
                       for k, i in batch['batch_turn_info'].items()}
    # y_dict_dataloader = {k: batch['y_dict'][k].data.cpu().numpy() for k in [ 'UT']}
    y_dict_dataloader = {k: batch['y_dict']
                         [k].data.cpu().numpy() for k in ['NT']}
    rest_batch = {k: batch[k].data.cpu().numpy() for k in batch_keys}
    batch_dict = {**batch_turn_info, **y_dict_dataloader, **rest_batch}
    return batch_dict


def analyze_error(data, update_annots_test, test_dataloader):
    error_data_types = ['loss', 'prediction_frame_count', 'true_offset',
                        'sys_full_overlap', 'usr_turn_end_da_nite', 'sys_turn_strt_da_nite']
    error_data_types += ['file', 'a_usr', 'usr_trn_strt_t', 'usr_trn_end_t',
                         'sys_trn_strt_t', 'sys_trn_end_t', 'usr_turn_idx', 'sys_trn_idx']
    error_data = {o: [] for o in error_data_types}
    for dat in data:
        analysis_loss_per_batch, analysis_count, y_trgt_out, batch = dat
        files = [test_dataloader.dataset.file_list[f]
                 for f in batch['file_idx']]
        a_usrs = [['A', 'B'][a] for a in batch['a_idx']]
        usr_update_idxs = batch['ass_usr_turn']
        sys_trn_idxs = batch['update_idx']
        sel_batch_indices = np.where(~(np.array(usr_update_idxs) == -1))[0]
        for sel_y_trgt, sel_bat in enumerate(sel_batch_indices):
            file, a_usr, usr_turn_idx, sys_trn_idx = files[sel_bat], a_usrs[
                sel_bat], usr_update_idxs[sel_bat], sys_trn_idxs[sel_bat]
            # update_strt_f, update_end_f = batch['update_strt_f'][sel_bat][0], batch['update_end_f'][sel_bat][0]
            # update_strt_t, update_end_t = batch['update_strt_t'][sel_bat][0], batch['update_end_t'][sel_bat][0]
            # ass_usr_ipu_strt_f = batch['associated_usr_ipu_strt_f'][sel_bat][0]
            b_sys = [a_b for a_b in ['A', 'B'] if not (a_usr == a_b)][0]

            # Collect necessary info
            usr_trn_strt_t = update_annots_test[file][a_usr]['turn_words_start_time'][usr_turn_idx]
            usr_trn_end_t = update_annots_test[file][a_usr]['turn_words_end_time'][usr_turn_idx]
            sys_trn_strt_t = update_annots_test[file][b_sys]['turn_words_start_time'][sys_trn_idx]
            sys_trn_end_t = update_annots_test[file][b_sys]['turn_words_end_time'][sys_trn_idx]
            sys_turn_full_overlap = update_annots_test[file][b_sys]['turn_full_overlap'][sys_trn_idx]

            usr_turn_end_da_nite = update_annots_test[file][a_usr]['turn_words_end_da_nite'][usr_turn_idx]
            sys_turn_strt_da_nite = update_annots_test[file][b_sys]['turn_words_start_da_nite'][sys_trn_idx]

            true_offset = sys_trn_strt_t - usr_trn_end_t
            error_data['loss'].append(
                float(analysis_loss_per_batch[sel_y_trgt]))
            error_data['prediction_frame_count'].append(
                int(analysis_count[sel_y_trgt]))
            error_data['true_offset'].append(true_offset)
            error_data['file'].append(file)
            error_data['a_usr'].append(a_usr)
            error_data['usr_trn_strt_t'].append(float(usr_trn_strt_t))
            error_data['usr_trn_end_t'].append(float(usr_trn_end_t))
            error_data['sys_trn_strt_t'].append(float(sys_trn_strt_t))
            error_data['sys_trn_end_t'].append(float(sys_trn_end_t))
            error_data['usr_turn_idx'].append(int(usr_turn_idx))
            error_data['sys_trn_idx'].append(int(sys_trn_idx))
            error_data['sys_full_overlap'].append(int(sys_turn_full_overlap))
            error_data['usr_turn_end_da_nite'].append(usr_turn_end_da_nite)
            error_data['sys_turn_strt_da_nite'].append(sys_turn_strt_da_nite)

    json.dump(error_data, open(
        naming_dict['fold_name'] + '/error_data.json', 'w'), indent=4)


def test(model, input_dataloader, full_test_flag, results_dict, iteration, epoch):
    model.eval()
    num_feat_per_person = model.module.num_feat_per_person
    test_str = input_dataloader.dataset.set_type
    num_test_batches = len(input_dataloader)
    loss_dict_test = {task: torch.tensor(
        0.0) for task in pred_task_dict['active_outputs']}
    loss_dict_test['all'] = torch.tensor(0.0)
    num_pred_samples_for_result = {
        task: 0 for task in pred_task_dict['active_outputs']}
    model.module.reset_hidden(test_str)

    # hidden_inference = model.module.hidden_inference[test_str]
    file_list = input_dataloader.dataset.file_list
    input_update_annots = input_dataloader.dataset.update_annots_test
    stats_data, error_data, sanity_check_data, rand_strt_f_dict_list = [], [], [], []
    test_time = t.time()
    batch_time = t.time()
    for batch_ndx, batch in enumerate(input_dataloader):

        mod_in = {k: v for k, v in batch.items() if not (k in ['y_dict'])}
        cont_file_indx, cont_ab_indx = batch['file_idx'], batch['a_idx']
        # h_inf = hidden_inference[:, cont_file_indx, cont_ab_indx, 0, :]
        # c_inf = hidden_inference[:, cont_file_indx, cont_ab_indx, 1, :]
        # mod_in['h_inf'] = h_inf.squeeze(0)
        # mod_in['c_inf'] = c_inf.squeeze(0)
        mod_in = {**batch['y_dict'], **mod_in}

        with torch.no_grad():
            bp_loss, outputs = model(**mod_in)

        # hidden_inference[:, cont_file_indx, cont_ab_indx, 0, :] = outputs['h_inf'].detach().cpu()
        # hidden_inference[:, cont_file_indx, cont_ab_indx, 1, :] = outputs['c_inf'].detach().cpu()
        if batch_ndx % 16 == 0 and test_str == 'test':

            print('test batch indx:'+str(batch_ndx)+'/'+str(num_test_batches))
            print('time taken:{:5.2f}'.format(t.time() - batch_time))
            batch_time = t.time()

        if lstm_sets_dict['analyze_error']:
            batch_dict = get_batch_items_for_full_test(batch)
            error_data.append([
                outputs['analysis_loss_per_batch'].data.cpu().numpy(),
                outputs['analysis_count'].data.cpu().numpy(),
                outputs['y_trgt_out'].data.cpu().numpy(),
                batch_dict
            ])

        if full_test_flag:
            probs_dict, y_trgt_stats_dict = {}, {}
            y_trgt_stats_dict['NT'] = outputs['y_trgt_out']['NT'].data.cpu(
            ).numpy()
            samp_out = outputs['samp_outs'].data.cpu().numpy()
            probs_dict['NT'] = outputs['prob_outs'].data.cpu().numpy()
            batch_dict = get_batch_items_for_full_test(batch)
            if model.module.lstm_sets_dict['encoder_settings']['skip_vae']:
                sys_enc_out_save = np.zeros(samp_out.shape[0])
                sys_enc_mu_save = np.zeros(samp_out.shape[0])
                sys_enc_log_var_save = np.zeros(samp_out.shape[0])
                rand_strt_dict_save = {k: d.data.cpu().numpy()
                                       for k, d in outputs['rand_strt_dict'].items()}
            else:
                sys_enc_out_save = outputs['sys_enc_out_save'].data.cpu(
                ).numpy()
                sys_enc_mu_save = outputs['sys_enc_mu'].data.cpu().numpy()
                sys_enc_log_var_save = outputs['sys_enc_log_var'].data.cpu(
                ).numpy()
                rand_strt_dict_save = {k: d.data.cpu().numpy()
                                       for k, d in outputs['rand_strt_dict'].items()}
            stats_data.append(
                (
                    probs_dict,
                    y_trgt_stats_dict,
                    samp_out,
                    batch_dict,
                    sys_enc_out_save,
                    sys_enc_mu_save,
                    sys_enc_log_var_save,
                    rand_strt_dict_save
                )
            )
            if lstm_sets_dict['sanity_check_bool']:
                rand_strt_f_dict_list.append(outputs['rand_strt_dict'])
            if batch_ndx % 64 == 1 and test_str == 'test':
                print('Plotting...')
                stats = get_test_stats(input_update_annots, stats_data,
                                       input_dataloader, file_list, epoch)
                print(model.module.lstm_sets_dict['just_test_folder'])
                print(naming_dict['fold_name'])

        loss_dict_test = {k: float(loss_dict_test[k]) + torch.sum(
            v.data.cpu()).float() for k, v in outputs['loss_dict_train_raw'].items()}
        num_pred_samples_for_result = {k: num_pred_samples_for_result[k] + torch.sum(
            v.data.cpu()).int() for k, v in outputs['num_pred_samples_for_result'].items()}
    sum_weights = 0
    sum_loss = 0
    for task in pred_task_dict['active_outputs']:
        sum_weights += pred_task_dict[task]['weight']
        sum_loss += loss_dict_test[task] * pred_task_dict[task]['weight'] * \
            1 / num_pred_samples_for_result[task]
        loss_dict_test[task] = loss_dict_test[task] * \
            1 / num_pred_samples_for_result[task]
    loss_dict_test['all'] = sum_loss/sum_weights

    if full_test_flag:
        if lstm_sets_dict['sanity_check_bool']:
            sanity_check_func(
                stats_data, rand_strt_f_dict_list, input_dataloader)

        stats = get_test_stats(input_update_annots, stats_data,
                               input_dataloader, file_list, epoch)
        results_dict[test_str]['stats'].append(stats)

    if lstm_sets_dict['analyze_error']:
        analyze_error(error_data, input_update_annots, input_dataloader)

    elapsed = t.time() - test_time
    loss_string = ''
    loss_string += ' '+test_str + ' | epoch {: 3d} | dur(s) {:5.2f} |'
    loss_string += ' '.join([task +
                             ' {:1.5f} |' for task in pred_task_dict['active_outputs']])
    loss_string += ' Weighted {:1.5f} '
    loss_string_items = [epoch, elapsed] + [
        loss_dict_test[task] for task in pred_task_dict['active_outputs']] + [loss_dict_test['all']]
    print('')
    print(loss_string.format(*loss_string_items))
    print('')
    results_dict[test_str]['all'].append(float(loss_dict_test['all']))
    for task in pred_task_dict['active_outputs']:
        results_dict[test_str][task].append(float(loss_dict_test[task]))
        loss_dict_test[task] = 0.0
    loss_dict_test['all'] = 0.0
    results_dict[test_str]['iteration'].append(int(iteration) + 1)
    results_dict[test_str]['epoch'].append(int(epoch))
    start_time = t.time()

    return results_dict[test_str]['all'], results_dict[test_str]['TL']
