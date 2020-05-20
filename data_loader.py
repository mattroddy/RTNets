import copy
import os
import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from util import get_file, get_update_annots

torch.set_default_dtype(torch.float32)


class ContDataset(Dataset):
    def __init__(self, dataset_settings_dict):
        self.batch_size = dataset_settings_dict['batch_size']
        self.pad_noise_bool = dataset_settings_dict['pad_noise_bool']

        # self.wrangle_features = wrangle_features
        self.feature_dict_list = dataset_settings_dict['feature_dict_list']
        self.pred_task_dict = dataset_settings_dict['pred_task_dict']
        self.file_list = dataset_settings_dict['file_list']
        self.file_to_idx = {f: i for i, f in enumerate(self.file_list)}
        self.ab_to_idx = {a: i for i, a in enumerate(['A', 'B'])}
        self.set_type = dataset_settings_dict['set_type']
        # self.prediction_length = self.pred_task_dict['VA']['pred_len']
        self.active_outputs = self.pred_task_dict['active_outputs']
        # self.data_select_list = dataset_settings_dict['data_select_list']
        # data_select_list = dataset_settings_dict['data_select_list']
        # self.g_f_str_to_idx = {g_f: idx for idx, g_f in enumerate(self.data_select_list)}
        self.device = dataset_settings_dict['device']
        # self.device_2 = dataset_settings_dict['device_2']
        self.output_order = dataset_settings_dict['output_order']
        self.use_ling = dataset_settings_dict['use_ling']
        # self.use_da_enc = dataset_settings_dict['use_da_enc']
        self.ling_timings = dataset_settings_dict['ling_timings']
        self.update_annots = dataset_settings_dict['update_annots']
        self.num_preloader_workers = dataset_settings_dict['num_preloader_workers']
        self.num_feat_per_person = dataset_settings_dict['num_feat_per_person']
        self.lstm_sets_dict = dataset_settings_dict['lstm_sets_dict']
        self.time_out_length = dataset_settings_dict['lstm_sets_dict']['time_out_length']
        self.extra_pad_frames = dataset_settings_dict['extra_pad_frames']
        self.pad_all_max_len_bool = dataset_settings_dict['lstm_sets_dict']['pad_all_max_len_bool']
        self.pad_max_len = dataset_settings_dict['lstm_sets_dict']['pad_max_len']
        # self.turn_batch = dataset_settings_dict['turn_batch']
        self.encoder_settings = dataset_settings_dict['lstm_sets_dict']['encoder_settings']
        self.unspec_tok = dataset_settings_dict['lstm_sets_dict']['unspec_tok']
        self.sil_tok = dataset_settings_dict['lstm_sets_dict']['sil_tok']
        self.wait_tok = dataset_settings_dict['lstm_sets_dict']['wait_tok']
        self.stop_tok = dataset_settings_dict['lstm_sets_dict']['stop_tok']
        self.pad_tok = dataset_settings_dict['lstm_sets_dict']['pad_tok']
        # self.dataset_settings_dict = dataset_settings_dict
        n_pre = self.lstm_sets_dict['pred_task_dict']['n_pre']
        if dataset_settings_dict['set_type'] == 'train' and self.lstm_sets_dict['two_sys_turn']:
            self.two_sys_turn = True
        else:
            self.two_sys_turn = False

        # Variables to fill
        self.len = 0
        self.dataset = []
        # self.file_update_lengths = []
        self.results_lengths = {}
        self.feature_size = 0
        # self.cont_hold_shift_points_count = 0
        # self.bc_points_count = 0
        self.use_saved_data_bool = dataset_settings_dict['use_saved_data_bool']
        self.use_saved_data_fold = dataset_settings_dict['use_saved_data_fold']
        self.sil_cov_matrices = {filename: {} for filename in self.file_list}
        self.sil_means = {filename: {} for filename in self.file_list}

        if self.use_ling:
            self.ling_size = 300
            self.tot_num_feat_per_person = self.num_feat_per_person['acous'] + \
                self.ling_size
        else:
            self.ling_size = 0
            self.tot_num_feat_per_person = self.num_feat_per_person['acous']

        self.files_ab = [
            [filename] + a_b for filename in self.file_list for a_b in [['A', 'B'], ['B', 'A']]]
        print('saved_data:' + self.use_saved_data_fold+self.set_type+'.p')
        if self.use_saved_data_bool and os.path.exists(self.use_saved_data_fold+self.set_type+'.p'):
            print('using saved data: ' + self.use_saved_data_fold)
            pdat = pickle.load(
                open(self.use_saved_data_fold + self.set_type + '.p', 'rb'))
            self.files_ab = pdat['files_ab']
            self.dataset = pdat['dataset']
            self.sil_means = pdat['sil_means']
            self.sil_cov_matrices = pdat['sil_cov_matrices']
        else:
            data_f_dict, data_g_dict = {}, {}
            args = []
            for file_name in self.file_list:
                file_args = [file_name, dataset_settings_dict]
                args.append(file_args)

            mult_out = []
            for arg in args:
                get_file_out = get_file(arg)
                mult_out.append(get_file_out)

            for mult in mult_out:
                f, g, set_obj_train = mult
                data_f_dict.update(f)
                data_g_dict.update(g)

            data_dict = {'A': data_f_dict,
                         'B': data_g_dict}

            # get annotations
            list_in = []
            for f_idx, lst in enumerate(self.files_ab):
                file, a_usr, b_sys = lst
                annots_usr = self.update_annots[file][a_usr]
                annots_sys = self.update_annots[file][b_sys]
                data_usr = data_dict[a_usr][file]['acous']
                data_sys = data_dict[b_sys][file]['acous']
                num_feat_per_person = self.num_feat_per_person['acous']
                pad_noise_bool = self.pad_noise_bool
                list_in.append([f_idx, file, a_usr, b_sys, annots_usr, annots_sys,
                                data_usr, data_sys, num_feat_per_person, pad_noise_bool, n_pre])
            print('reached pre multiprocessing in dataloader')

            # get updates
            new_files_ab = []
            data_strt_idx = 0
            for f_idx, ls in enumerate(list_in):
                data_pts, sil_means, cov, file, a_usr, b_sys, file_length = get_update_annots(
                    ls)
                for data_pt in data_pts:
                    self.dataset.append(data_pt)
                # self.dataset.extend(data_pts)
                # note mean and cov for opposite speaker's files used
                self.sil_means[file][b_sys] = sil_means
                self.sil_cov_matrices[file][b_sys] = cov
                new_files_ab.append([f_idx, file[0], a_usr[0], b_sys[0],
                                     file_length, data_strt_idx, data_strt_idx+file_length])
                data_strt_idx += file_length

            self.files_ab = new_files_ab

            # Save data
            if self.use_saved_data_bool and not os.path.exists(self.use_saved_data_fold + self.set_type + '.p'):
                print('Saving DATA: '+self.set_type)
                if not os.path.exists(self.use_saved_data_fold):
                    os.makedirs(self.use_saved_data_fold)
                pickle.dump(
                    {
                        'dataset': self.dataset,
                        'files_ab': self.files_ab,
                        'sil_cov_matrices': self.sil_cov_matrices,
                        'sil_means': self.sil_means
                    },
                    open(self.use_saved_data_fold+self.set_type+'.p', 'wb')
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if not idx == -1:
            if self.two_sys_turn:
                item = [self.dataset[idx-1], self.dataset[idx]]
                item[0]['dataset_idx'] = [idx-1]
                item[1]['dataset_idx'] = [idx]
            else:
                item = self.dataset[idx]
                item['dataset_idx'] = [idx]
        else:
            item = []
        return item

    def collate_fn(self, batch):
        if self.two_sys_turn:
            batch_tmp = []
            for bat in batch:
                batch_tmp.append(bat[0])
                batch_tmp.append(bat[1])
            batch = batch_tmp
            del(batch_tmp)
        else:
            batch = [bat for bat in batch if len(bat)]

        def get_seq_len(seq):
            if len(seq.shape) == 2:
                length = seq.shape[0]
            elif len(seq.shape) == 1:
                length = 0
            else:
                raise Exception('bad shape')
            return length

        # use this for y_trt_t(0), file(5), a_usr(6), usr_update_idx(7) sys_trn_idx(8), cont_pred_vecs(9), cont_hold_shifts(10)
        def get_basic_list_item(batch, i):
            item = [batch[b][i] for b in range(len(batch))]
            return item

        def get_update_item_list_and_ling(batch, indx, lengths):
            items, y_LM_items, y_NT_items = [], [], []
            for b in range(len(batch)):
                if lengths[b] > 0:
                    if self.use_ling:
                        emb_inds_words = []
                        for wrd in batch[b]['word_da_dict'][indx + '_words']:
                            emb_inds_words.append(torch.LongTensor(
                                [self.nlp.vocab.vectors.key2row[self.nlp.vocab.strings[wrd]]]))
                        emb_inds = [torch.LongTensor(
                            [self.sil_tok])] * (len(batch[b][indx])+1)
                        y_NT_item = [torch.LongTensor(
                            [0])] * (len(batch[b][indx])+1)
                        for i in range(len(batch[b]['word_da_dict'][indx+'_word_strt_frames'])):
                            if self.ling_timings['updates'] == 'held':
                                # advance the annotations for ASR
                                wrd_strt = batch[b]['word_da_dict'][indx +
                                                                    '_word_strt_frames'][i]
                                wrd_end = batch[b]['word_da_dict'][indx +
                                                                   '_word_end_frames'][i]
                            elif self.ling_timings['updates'] == 'one_shot':
                                wrd_strt = batch[b]['word_da_dict'][indx +
                                                                    '_word_end_frames'][i]
                                wrd_end = wrd_strt + 1
                            emb_inds[wrd_strt: wrd_end] = [
                                emb_inds_words[i]] * (wrd_end - wrd_strt)
                            if indx == 'sys_update':
                                y_NT_item[wrd_strt] = 1

                        padded_inds = torch.nn.utils.rnn.pad_sequence(
                            emb_inds, padding_value=self.pad_tok).transpose(0, 1)
                        # !! note: only using first token for the moment (need to create new ones later)
                        y_LM_emb_inds = padded_inds[:, 0]
                        y_LM_emb_inds[:batch[b]['word_da_dict']
                                      [indx + '_word_strt_frames'][0]] = self.wait_tok
                        y_LM_emb_inds[batch[b]['word_da_dict']
                                      [indx + '_word_end_frames'][-1]:] = self.stop_tok
                        if indx == 'sys_update':
                            y_NT_item[batch[b]['word_da_dict']
                                      [indx + '_word_end_frames'][-1]] = 1
                        update_strt_f = batch[b]['update_strt_f'][0]
                        ass_ipu_strt_f = batch[b]['associated_usr_ipu_strt_f'][0]
                        rand_strt_f = max([0, ass_ipu_strt_f])
                        # Need to average the glove vectors when there is more than one token in a word
                        ling_feats = (torch.sum(self.embeds_sys(self.embeds(
                            padded_inds.to(self.embeds.weight.device))), dim=1).t()).t()
                        y_LM_emb_inds[:rand_strt_f] = -1
                        stacked_feats = torch.cat([ling_feats[:len(batch[b][indx])].to(
                            self.device), torch.tensor(batch[b][indx]).to(self.device)], dim=1)
                        items.append(stacked_feats)
                        y_LM_items.append(y_LM_emb_inds)
                        y_NT_items.append(torch.LongTensor(y_NT_item))
                    else:
                        y_LM_items = []
                        y_NT_items = []
                        items.append(torch.tensor(batch[b][indx]))
            return items, y_LM_items, y_NT_items

        def get_updates(batch, test_seq_len):
            # pack_seq_indices = ['usr_update', 'sys_update']
            pack_seq_indices = ['sys_update']
            selection_vecs_for_pad = []
            output, output_y_LM = [], []

            for indx in pack_seq_indices:
                lengths = np.array([get_seq_len(batch[b][indx])
                                    for b in range(len(batch))])
                selection_vecs_for_pad.append(lengths)
                items, y_LM_items, y_NT_items = get_update_item_list_and_ling(
                    batch, indx, lengths)
                padded_seq = pad_sequence(items)

                if indx == 'sys_update':
                    if self.two_sys_turn:
                        y_LM_tmp, y_NT_tmp = [], []
                        for i in range(1, len(y_LM_items), 2):
                            y_LM_tmp.append(
                                torch.cat([y_LM_items[i - 1], y_LM_items[i]], dim=0))
                            y_NT_tmp.append(
                                torch.cat([y_NT_items[i - 1], y_NT_items[i]], dim=0))
                        y_LM_items = y_LM_tmp
                        y_NT_items = y_NT_tmp
                        del y_LM_tmp
                        del y_NT_tmp

                    padded_y_LM = pad_sequence(y_LM_items, padding_value=-1)
                    padded_y_NT = pad_sequence(y_NT_items, padding_value=-1)
                    # pad_val = self.sil_tok + 2 if self.lstm_sets_dict['use_wait_stop_tok'] else self.sil_tok
                    pad_val = self.stop_tok
                    prev_word_embeds_inds = pad_sequence(
                        y_LM_items, padding_value=pad_val)
                    prev_word_embeds_inds = torch.cat(
                        [
                            prev_word_embeds_inds,
                            pad_val*torch.ones([test_seq_len - prev_word_embeds_inds.shape[0],
                                                prev_word_embeds_inds.shape[1]], dtype=torch.long)
                        ]
                    )
                    # advance word token and pad to the length of the test sequence
                    y_LM = padded_y_LM[1:]
                    y_LM = torch.cat(
                        [y_LM, -1 * torch.ones([test_seq_len - y_LM.shape[0], y_LM.shape[1]], dtype=torch.long)])
                    y_NT = padded_y_NT[1:]
                    y_NT = torch.cat(
                        [y_NT, -1 * torch.ones([test_seq_len - y_NT.shape[0], y_NT.shape[1]], dtype=torch.long)]).float()
            return [], [], [], [], selection_vecs_for_pad, y_LM.data.numpy(), prev_word_embeds_inds, y_NT.data.numpy()

        def get_sys_trns(batch, test_seq_len):
            batch_trn_indices, num_turns_per_batch, items_0, items_1 = [], [], [], []
            lengths_acous, lengths_ling, sys_batch_lens = [], [], []
            sys_enc_strt_list, sys_enc_end_list = [], []
            word_times_list, word_tok_list, val_tok_list = [], [], []
            max_len, bat_strt_indx = 0, 0
            for b in range(len(batch)):
                num_turns_per_batch.append(len(batch[b]['sys_trn']))
                sys_batch_lens.append([])
                for seq in batch[b]['sys_trn']:
                    max_len = max([seq.shape[0], max_len])
                    lengths_acous.append(seq.shape[0])
                    sys_batch_lens[-1].append(seq.shape[0])
                    # Not currently set up for multiple turn_batches

                    # return stacked acoustic and linguistic features
                    emb_inds_words = []
                    for wrd in batch[b]['word_da_dict']['sys_enc_words']:
                        emb_inds_words.append(torch.LongTensor(
                            [self.nlp.vocab.vectors.key2row[self.nlp.vocab.strings[wrd]]]))
                    lst = batch[b]['word_da_dict']['sys_enc_words']
                    enc_strt_frames = copy.copy(
                        batch[b]['word_da_dict']['sys_enc_word_strt_frames'])
                    enc_end_frames = copy.copy(
                        batch[b]['word_da_dict']['sys_enc_word_end_frames'])
                    word_times = [[0, 0]]
                    emb_lengths = [1]
                    tok_vals = [torch.LongTensor([self.wait_tok])]
                    tok_words = [torch.LongTensor([self.wait_tok])]
                    for i in range(len(enc_strt_frames)):
                        if enc_strt_frames[i] == enc_end_frames[i]:
                            enc_end_frames[i] = enc_end_frames[i] + 1

                        if enc_strt_frames[i] < word_times[-1][1]:
                            enc_strt_frames[i] = word_times[-1][1]

                        if enc_strt_frames[i] != word_times[-1][1]:
                            word_times.append(
                                [word_times[-1][1], enc_strt_frames[i]])
                            tok_vals.append(torch.LongTensor([self.sil_tok]))
                            tok_words.append(torch.LongTensor([self.sil_tok]))
                            emb_lengths.append(1)
                        # hack to deal with a problem with file 'sw2361' (two tokens: [laughter-you],[laughter-know] with length 0)
                        if enc_end_frames[i] < enc_strt_frames[i]:
                            word_times.append(
                                [enc_end_frames[i], enc_strt_frames[i]])
                        else:
                            word_times.append(
                                [enc_strt_frames[i], enc_end_frames[i]])
                        # end hack

                        # To get longer than one token use this
                        tok_vals.append(emb_inds_words[i])
                        emb_lengths.append(len(emb_inds_words[i]))
                        tok_words.append(lst[i])

                    # another hack to deal with 'sw2361'
                    if word_times[-1][0] == word_times[-1][1]:
                        word_times.pop()
                        tok_vals.pop()
                        emb_lengths.pop()
                        tok_words.pop()

                    # replace wait token times, and add stop token
                    word_times[0] = [-1, -1]
                    word_times.append([-2, -2])
                    tok_vals.append(torch.LongTensor([self.stop_tok]))
                    tok_words.append(torch.LongTensor([self.stop_tok]))
                    emb_lengths.append(1)
                    padded_inds = torch.nn.utils.rnn.pad_sequence(
                        tok_vals, padding_value=self.pad_tok).transpose(0, 1)
                    # hack to only use the first token in the word
                    padded_inds = padded_inds[:, 0]
                    # end hack

                    ling_feats = self.embeds_sys(self.embeds(
                        padded_inds.to(self.embeds.weight.device)))

                    lengths_ling.append(ling_feats.shape[0])
                    items_1.append(ling_feats)
                    items_0.append(torch.tensor(seq))
                    word_times_list.append(word_times)
                    word_tok_list.append(tok_words)
                    val_tok_list.append(padded_inds)

                batch_trn_indices.append(
                    (bat_strt_indx, bat_strt_indx + len(batch[b]['sys_trn'])))
                bat_strt_indx = bat_strt_indx + len(batch[b]['sys_trn'])
                sys_enc_strt_list.append(torch.LongTensor(
                    batch[b]['word_da_dict']['sys_enc_word_strt_frames']))
                sys_enc_end_list.append(torch.LongTensor(
                    batch[b]['word_da_dict']['sys_enc_word_end_frames']))

            padded_seq_0 = pad_sequence(items_0).to(self.device)
            lengths_0 = lengths_acous

            padded_seq_1 = pad_sequence(items_1).to(self.device)
            lengths_1 = lengths_ling

            sys_enc_end_lengths = torch.LongTensor(
                [l.shape[0] for l in sys_enc_end_list])
            sys_enc_end_list_pad = pad_sequence(
                sys_enc_end_list, batch_first=True)
            sys_enc_strt_lengths = torch.LongTensor(
                [l.shape[0] for l in sys_enc_strt_list])
            sys_enc_strt_list_pad = pad_sequence(
                sys_enc_strt_list, batch_first=True)
            val_tok_lengths = torch.LongTensor(
                [l.shape[0] for l in val_tok_list])
            val_tok_list_pad = pad_sequence(val_tok_list, batch_first=True)
            wrd_times_strt_list = [torch.LongTensor(
                [wrd[0] for wrd in b]) for b in word_times_list]
            wrd_times_strt = pad_sequence(
                wrd_times_strt_list, batch_first=True)
            wrd_times_end_list = [torch.LongTensor(
                [wrd[1] for wrd in b]) for b in word_times_list]
            wrd_times_end_lengths = torch.LongTensor(
                [l.shape[0] for l in wrd_times_end_list])
            wrd_times_end = pad_sequence(wrd_times_end_list, batch_first=True)

            batch_trn_info = {
                'num_turns_per_batch': torch.LongTensor(num_turns_per_batch),
                'sys_batch_lens': torch.LongTensor(sys_batch_lens),
                'sys_enc_word_starts': sys_enc_strt_list_pad,
                'sys_enc_word_starts_lengths': sys_enc_strt_lengths,
                'sys_enc_word_ends': sys_enc_end_list_pad,
                'sys_enc_word_ends_lengths': sys_enc_end_lengths,
                'word_times_strt_list_sil': wrd_times_strt,
                'word_times_end_list_sil': wrd_times_end,
                'word_times_lengths': wrd_times_end_lengths,
                'val_tok_list_sil': val_tok_list_pad,
                'val_tok_list_sil_lengths': val_tok_lengths
            }

            # Get new y_NT that includes the silence tokens
            y_NT_with_sil_list = []
            # num_batches,turn_batch_num = int(len(batch)/2) ,1 if self.two_sys_turn else len(batch), 0
            for b in range(len(batch)):
                y_NT_with_sil_idxs_strts = wrd_times_strt_list[b][1:-1] + int(
                    batch[b]['word_da_dict']['sys_update_word_strt_frames'][0]) - 1  # advanced
                y_NT_with_sil_idxs_ends = wrd_times_end_list[b][1:-1] + int(
                    batch[b]['word_da_dict']['sys_update_word_strt_frames'][0]) - 1  # advanced
                y_NT_with_sil_item = torch.zeros(
                    batch[b]['sys_update'].shape[0]) if self.two_sys_turn else torch.zeros(test_seq_len)
                for wrd_idx in y_NT_with_sil_idxs_strts:
                    y_NT_with_sil_item[wrd_idx] = 1
                # also annotate the stop_tok
                y_NT_with_sil_item[y_NT_with_sil_idxs_ends[-1]] = 1
                y_NT_with_sil_item[y_NT_with_sil_idxs_ends[-1]+1:] = -1
                y_NT_with_sil_list.append(y_NT_with_sil_item)

            if self.two_sys_turn:
                y_NT_list_tmp = []
                for b_i in range(1, len(batch), 2):
                    y_NT_list_tmp.append(
                        torch.cat([y_NT_with_sil_list[b_i - 1], y_NT_with_sil_list[b_i]]))
                y_NT_with_sil_list = y_NT_list_tmp
                del y_NT_list_tmp

            y_NT_with_sil = pad_sequence(y_NT_with_sil_list, padding_value=-1)

            return padded_seq_0, padded_seq_1, lengths_0, lengths_1, batch_trn_info, y_NT_with_sil.numpy()

        # For the context vectors, they should be initialized to zero for the batch size at the beginning.
        # We then have a file attached to each of the context elements until the file is finished.
        # Then check whether it is the first update and use that to update the context.

        # Use this to get the test sequence
        def get_test_seq(batch):
            selection_vecs_test = []
            lengths = np.array([len(batch[b]['test_seq'])
                                for b in range(len(batch))])
            selection_vecs_test.append(lengths)
            if all(lengths == 0):
                out_test_seq = []
                return out_test_seq, selection_vecs_test
            if self.pad_all_max_len_bool:
                pad_length = self.pad_max_len
            else:
                # need to use y_ut_max because of overlap
                y_UT_max_length = [len(batch[b]['y_UT'])
                                   for b in range(len(batch))]
                pad_length = max(np.concatenate(
                    [lengths, y_UT_max_length])) + self.extra_pad_frames
            sel_indices = np.where(lengths > 0)[0]
            files = [batch[int(b)]['file'] for b in sel_indices]
            a_b_usrs = [batch[int(b)]['a_usr'] for b in sel_indices]
            padded_items = []
            for b, file, a_b in zip(sel_indices, files, a_b_usrs):
                if (len(batch[b]['test_seq']) < pad_length) & ~ np.array(batch[b]['test_seq'] == [-1]).all() and not(self.two_sys_turn):
                    if self.pad_noise_bool:
                        pad_vals = np.random.multivariate_normal(
                            self.sil_means[file[0]][a_b[0]],
                            self.sil_cov_matrices[file[0]][a_b[0]],
                            size=pad_length - len(batch[b]['test_seq']))
                    else:
                        pad_vals = np.ones([pad_length - len(batch[b]['test_seq']), batch[b]
                                            ['test_seq'].shape[1]]) * self.sil_means[file[0]][a_b[0]]
                    padded_item = np.concatenate(
                        [batch[b]['test_seq'], pad_vals]).astype(np.float32)
                else:
                    padded_item = batch[b]['usr_update'].astype(np.float32)

                # get word features and advance
                if self.use_ling:
                    emb_inds_words = []
                    for wrd in batch[b]['word_da_dict']['test_words']:
                        emb_inds_words.append(torch.LongTensor(
                            [self.nlp.vocab.vectors.key2row[self.nlp.vocab.strings[wrd]]]))
                    emb_inds = torch.LongTensor(
                        [self.sil_tok] * len(padded_item))
                    for i in range(len(batch[b]['word_da_dict']['test_word_strt_frames'])):
                        if self.ling_timings['inference'] == 'held':
                            # advance the annotations for ASR
                            wrd_length = batch[b]['word_da_dict']['test_word_end_frames'][i] - \
                                batch[b]['word_da_dict']['test_word_strt_frames'][i]
                            # advance the annotations for ASR (100ms), 1 is correct for 100ms
                            wrd_strt = batch[b]['word_da_dict']['test_word_end_frames'][i] + 1
                            wrd_end = wrd_strt + wrd_length
                            emb_inds[wrd_strt:wrd_end] = [
                                emb_inds_words[i]]*(wrd_end-wrd_strt)
                        elif self.ling_timings['inference'] == 'one_shot':
                            # advance the annotations for ASR
                            wrd_strt = batch[b]['word_da_dict']['test_word_end_frames'][i] + 1
                            emb_inds[wrd_strt] = emb_inds_words[i]
                        elif self.ling_timings['inference'] == 'one_shot_unspec_held':
                            unspec_strt = batch[b]['word_da_dict']['test_word_strt_frames'][i] + 2
                            unspec_end = batch[b]['word_da_dict']['test_word_end_frames'][i] + 1
                            emb_inds[unspec_strt:unspec_end] = self.unspec_tok
                            if self.two_sys_turn:
                                # advance the annotations for ASR
                                wrd_strt = batch[b]['word_da_dict']['test_word_end_frames'][i] + 1
                                wrd_strt = min(wrd_strt, len(emb_inds)-1)
                                emb_inds[wrd_strt] = emb_inds_words[i]
                            else:
                                # advance the annotations for ASR
                                wrd_strt = batch[b]['word_da_dict']['test_word_end_frames'][i] + 1
                                emb_inds[wrd_strt] = emb_inds_words[i]

                    padded_inds = emb_inds
                    ling_feats = self.embeds_usr(self.embeds(
                        padded_inds.to(self.embeds.weight.device)))
                    stacked_feats = torch.cat([ling_feats[:len(padded_item)].to(
                        self.device), torch.tensor(padded_item).to(self.device)], dim=1)
                else:
                    stacked_feats = torch.tensor(padded_item)
                padded_items.append(stacked_feats)
            if self.two_sys_turn:
                padded_tmp = []
                lengths_tmp = []
                for i in range(1, len(padded_items), 2):
                    padded_tmp.append(
                        torch.cat([padded_items[i - 1], padded_items[i]], dim=0))
                    lengths_tmp.append(len(padded_tmp[-1]))
                stacked_items = pad_sequence(padded_tmp)
                lengths = torch.tensor(lengths_tmp)
                del (padded_tmp)
                test_length = len(stacked_items)
            else:
                stacked_items = torch.stack(padded_items, dim=1)
                test_length = len(padded_item)
            stacked_lengths = lengths
            return stacked_items, stacked_lengths, selection_vecs_test, test_length

        # Get the items
        y_strt_f, file, a_usr = get_basic_list_item(batch, 'y_strt_f'), get_basic_list_item(
            batch, 'file'), get_basic_list_item(batch, 'a_usr')
        update_strt_f, update_end_f = get_basic_list_item(
            batch, 'update_strt_f'), get_basic_list_item(batch, 'update_end_f')
        update_strt_t, update_end_t = get_basic_list_item(
            batch, 'update_strt_t'), get_basic_list_item(batch, 'update_end_t')
        associated_usr_ipu_strt_f = get_basic_list_item(
            batch, 'associated_usr_ipu_strt_f')
        associated_usr_ipu_end_f = get_basic_list_item(
            batch, 'associated_usr_ipu_end_f')
        y_end_f, y_length = get_basic_list_item(
            batch, 'y_end_f'), get_basic_list_item(batch, 'y_length')
        ass_usr_turn, update_idx = get_basic_list_item(
            batch, 'associated_usr_turn'), get_basic_list_item(batch, 'update_idx')
        y_strt_t, y_length = get_basic_list_item(
            batch, 'y_strt_t'), get_basic_list_item(batch, 'y_length')
        dataset_idx = get_basic_list_item(batch, 'dataset_idx')
        # usr_update, sys_update, selection_vecs_for_pad,y_LM = get_updates(batch)

        test_seq, test_lengths, selection_vecs_for_test_seq, test_seq_len = get_test_seq(
            batch)
        sys_trn_0, sys_trn_1, lengths_0, lengths_1, batch_turn_info, y_NT_with_sil = get_sys_trns(
            batch, test_seq_len)
        file_idx = [self.file_to_idx[f[0]] for f in file]
        a_idx = [self.ab_to_idx[a_b[0]] for a_b in a_usr]

        output = {'y_strt_f': torch.LongTensor(y_strt_f),
                  'y_strt_t': torch.FloatTensor(y_strt_t),
                  'y_end_f': torch.LongTensor(y_end_f),
                  'y_length': torch.LongTensor(y_length),
                  'associated_usr_ipu_strt_f': torch.LongTensor(associated_usr_ipu_strt_f),
                  'associated_usr_ipu_end_f': torch.LongTensor(associated_usr_ipu_end_f),
                  'dataset_idx': torch.LongTensor(dataset_idx),
                  'y_dict': {
            # 'UT': torch.FloatTensor(get_y_UT_HS('y_UT')).transpose(0,1),
            'NT': torch.FloatTensor(y_NT_with_sil).transpose(0, 1)
        },
            # acoustic or combined feats
            'sys_trn_0': sys_trn_0.transpose(0, 1).cpu(),
            'sys_trn_0_lengths': torch.LongTensor(lengths_0),
            # just ling feats
            'sys_trn_1': sys_trn_1.transpose(0, 1).cpu(),
            'sys_trn_1_lengths': torch.LongTensor(lengths_1),
            'batch_turn_info': batch_turn_info,
            'test_seq': test_seq.transpose(0, 1).cpu(),
            'test_lengths': torch.LongTensor(test_lengths),
            'update_strt_f': torch.LongTensor(update_strt_f),
            'update_end_f': torch.LongTensor(update_end_f),
            'update_strt_t': torch.FloatTensor(update_strt_t),
            'update_end_t': torch.FloatTensor(update_end_t),
            'update_idx': torch.LongTensor(update_idx),
            'ass_usr_turn': torch.LongTensor(ass_usr_turn),
            'file_idx': torch.LongTensor(file_idx),
            'a_idx': torch.LongTensor(a_idx)
        }

        return (output)


class MySampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = self.dataset.batch_size
        self.my_files_ab = copy.copy(self.dataset.files_ab)
        self.my_indices, self.my_indices_no_first = self.shuffle()

    # @property
    def shuffle(self):
        self.files_ab = self.dataset.files_ab
        files_copy = copy.deepcopy(self.files_ab)
        random.shuffle(files_copy)

        loc_bsize = self.batch_size
        my_indices, my_indices_no_first = [], []
        # append a counter that indexes the positions of the data in the dataset list
        for i in range(len(files_copy)):
            files_copy[i].append(files_copy[i][-2])

        selected_files = files_copy[:self.batch_size]
        files_copy[:self.batch_size] = []
        check_counter = 0
        b_i = 0
        stop_bool = False
        while not stop_bool:
            if (b_i > len(selected_files)-1) or (selected_files[b_i][-1] >= selected_files[b_i][-2]):
                if len(files_copy):
                    selected_files[b_i] = files_copy.pop()
                    my_indices.append(selected_files[b_i][-1])
                    if selected_files[b_i][-1] > selected_files[b_i][-3]:
                        my_indices_no_first.append(selected_files[b_i][-1])
                    selected_files[b_i][-1] += 1
                    check_counter += 1
                else:
                    my_indices.append(-1)
            else:
                my_indices.append(selected_files[b_i][-1])
                if selected_files[b_i][-1] > selected_files[b_i][-3]:
                    my_indices_no_first.append(selected_files[b_i][-1])
                selected_files[b_i][-1] += 1
                check_counter += 1
            b_i += 1
            if b_i >= loc_bsize:
                b_i = 0
            stop_bool = all([n[-1] == n[-2]
                             for n in selected_files]) and len(files_copy) == 0
        return my_indices, my_indices_no_first

    def __iter__(self):
        self.my_indices, self.my_indices_no_first = self.shuffle()
        return iter(self.my_indices)

    def __len__(self):
        return len(self.dataset)
