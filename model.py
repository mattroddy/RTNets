# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

framelen = 0.05
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

SET_TRAIN = 0
SET_VALID = 1
SET_TEST = 2

SET_DICT = {
    'train': 0,
    'valid': 1,
    'test': 2
}


class ContModel(nn.Module):
    def __init__(self, num_feat_per_person, lstm_sets_dict, device, context_vec_settings, embeds_usr, embeds_sys, embeds):
        super(ContModel, self).__init__()

        self.lstm_sets_dict = lstm_sets_dict
        self.autoregress = lstm_sets_dict['train_autoregress']
        self.pred_task_dict = lstm_sets_dict['pred_task_dict']
        self.active_outputs = self.pred_task_dict['active_outputs']
        self.temperature = lstm_sets_dict['temperature']
        self.lstm_sets_dict = lstm_sets_dict
        self.context_vec_settings = context_vec_settings
        self.embeds_usr = embeds_usr
        self.embeds_sys = embeds_sys
        self.embeds = embeds
        self.num_feat_per_person = num_feat_per_person
        self.encoder_settings = lstm_sets_dict['encoder_settings']
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_func_BCE_logit = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_func_BCE = nn.BCELoss(reduction='sum')
        self.ling_dim_size = 300

        if lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
            self.acous_dim_size = 0
            self.ling_dim_size = 0
            inf_lstm_input_size = 1
        else:
            self.acous_dim_size = self.num_feat_per_person - self.ling_dim_size
            inf_lstm_input_size = 2 * \
                lstm_sets_dict['master_encoder_hidden_size'] + \
                self.num_feat_per_person
            w2_in_size = 2 * lstm_sets_dict['master_encoder_hidden_size']

        if lstm_sets_dict['response_encoder_hidden_size']:
            if not(lstm_sets_dict['encoder_settings']['use_acous']):
                self.response_0_dim = 0
                self.response_1_dim = self.ling_dim_size
                self.loc_response_encoder_hidden_size = lstm_sets_dict['response_encoder_hidden_size']
            elif not(lstm_sets_dict['encoder_settings']['use_ling']):
                self.response_0_dim = self.acous_dim_size
                self.response_1_dim = 0
                self.loc_response_encoder_hidden_size = lstm_sets_dict['response_encoder_hidden_size']
            else:
                self.response_0_dim = self.acous_dim_size
                self.response_1_dim = self.ling_dim_size
                self.loc_response_encoder_hidden_size = int(
                    lstm_sets_dict['response_encoder_hidden_size'] / 2)
                self.master_enc_in_dim = int(
                    lstm_sets_dict['response_encoder_hidden_size']) * 2

            if self.response_0_dim:
                self.response_encoder_lstm_0 = nn.LSTM(
                    self.response_0_dim,
                    self.loc_response_encoder_hidden_size,
                    bidirectional=True,
                    batch_first=True
                )
            if self.response_1_dim:
                self.response_encoder_lstm_1 = nn.LSTM(
                    self.response_1_dim,
                    self.loc_response_encoder_hidden_size,
                    bidirectional=True,
                    batch_first=True
                )

            self.master_encoder = nn.LSTM(
                self.loc_response_encoder_hidden_size*4,
                lstm_sets_dict['master_encoder_hidden_size'],
                bidirectional=True,
                batch_first=True
            )
            # inf_lstm_input_size += lstm_sets_dict['usr_encoder_hidden_size']
        else:
            inf_lstm_input_size += num_feat_per_person

        self.inference_lstm = nn.LSTM(
            inf_lstm_input_size,
            lstm_sets_dict['inference_hidden_size'],
            batch_first=True
        )

        self.init_output_layers()

        if not lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
            self.acous_embeds = nn.Embedding(
                2, self.loc_response_encoder_hidden_size*2)
            self.hidden_embeds = nn.Embedding(
                2, self.loc_response_encoder_hidden_size*4)

        self.acous_emb_wait_tok = 0
        self.acous_emb_stop_tok = 1
        self.hid_emb_stop_tok = 0
        self.hid_emb_none_tok = 1

        self.hidden_user, self.hidden_system = {}, {}
        self.hidden_user_encoder = {}
        # self.hidden_inference = {}
        # for k, i in context_vec_settings.items():
        # self.hidden_inference[k] = self.init_hidden(i, self.lstm_sets_dict['inference_hidden_size'])
        # if self.lstm_sets_dict['usr_encoder_hidden_size']:
        #   self.hidden_user_encoder[k] = self.init_hidden(i, self.lstm_sets_dict['usr_encoder_hidden_size'])
        # else:
        #   # this is lazy coding
        #   self.hidden_user_encoder[k] = self.init_hidden(i, 1)

        # VAE
        self.vae_setup()

    def vae_setup(self):
        if self.lstm_sets_dict['encoder_settings']['skip_vae']:
            self.wait_network = Reduce_Sys_Enc(
                6 * self.lstm_sets_dict['master_encoder_hidden_size'], 2 * self.lstm_sets_dict['master_encoder_hidden_size'])
        else:
            self.VAE = VAE(int(self.lstm_sets_dict['master_encoder_hidden_size'] * 6),
                           int(self.lstm_sets_dict['master_encoder_hidden_size']),
                           int(self.lstm_sets_dict['master_encoder_hidden_size'] / 2),
                           int(self.lstm_sets_dict['vae_dim'])
                           )
            self.Split_Wait_Start = Split_Wait_Start(self.lstm_sets_dict['vae_dim'],
                                                     2 * self.lstm_sets_dict['master_encoder_hidden_size'])

    # Hidden states
    def init_hidden(self, num_files, hidden_size):
        # [1,file_idx, ab_idx, hidden/cell, hidden_size]
        hidden = torch.zeros([1, num_files, 2, 2, hidden_size])
        return hidden

    def init_output_layers(self):
        out_dict = {}
        for task in self.active_outputs:
            # in_dim = self.lstm_sets_dict['inference_hidden_size']
            if not self.pred_task_dict[task]['bypass_layers']:
                in_dim = self.pred_task_dict[task]['in_dim']
                modules = []
                for _ in range(self.pred_task_dict[task]['output_layers']):
                    modules.append(
                        nn.Linear(in_dim, self.pred_task_dict[task]
                                  ['output_layer_size'])
                    )
                    modules.append(nn.ReLU())
                    in_dim = self.pred_task_dict[task]['output_layer_size']
                modules.append(
                    nn.Linear(in_dim, self.pred_task_dict[task]['pred_len']))
                out_dict[task] = nn.Sequential(*modules)
        self.out_dict = nn.ModuleDict(out_dict)

    def reset_hidden(self, set_type):
            # [1,file_idx, ab_idx, hidden/cell, hidden_size]
        num_files = self.context_vec_settings[set_type]
        # self.hidden_inference[SET_DICT[set_type]] = self.init_hidden(num_files, self.lstm_sets_dict['inference_hidden_size'])

    def forward(self, **kwargs):
        test_seq_sel_lengths = kwargs['test_lengths']
        cont_file_indx, cont_ab_indx = kwargs['file_idx'], kwargs['a_idx']
        batch_turn_info = kwargs['batch_turn_info']
        batch_size = kwargs['NT'].shape[0]
        if self.lstm_sets_dict['two_sys_turn'] and self.training:
            # for the system encodings
            encoding_batch_size = int(batch_size * 2)
        else:
            encoding_batch_size = batch_size
        # h_inf, c_inf = kwargs['h_inf'].unsqueeze(0), kwargs['c_inf'].unsqueeze(0)
        y_out = {task: [] for task in self.active_outputs}
        test_seq = kwargs['test_seq']
        test_seq_lengths = kwargs['test_lengths']

        word_times_strts = kwargs['batch_turn_info']['word_times_strt_list_sil']
        word_time_lengths = kwargs['batch_turn_info']['word_times_lengths']
        sys_trn_0 = kwargs['sys_trn_0']
        sys_trn_1 = kwargs['sys_trn_1']
        sys_trn_1_lengths = kwargs['sys_trn_1_lengths']

        # enc ablation stuff
        if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
            sys_trn_0.fill_(0)
            sys_trn_1.fill_(0)
        elif self.lstm_sets_dict['enc_ablation_setting'] == 'only_ling':
            sys_trn_0.fill_(0)
        elif self.lstm_sets_dict['enc_ablation_setting'] == 'only_acous':
            sys_trn_1.fill_(0)

        sys_trn_0 = pack_padded_sequence(
            sys_trn_0, kwargs['sys_trn_0_lengths'], batch_first=True, enforce_sorted=False)
        sys_trn_1 = pack_padded_sequence(
            sys_trn_1, sys_trn_1_lengths, batch_first=True, enforce_sorted=False)
        sel_batch = np.where(
            np.array(test_seq_sel_lengths.data.cpu().numpy() > 0))[0]
        y_trgt_dict = kwargs

        # Get user features encodings
        usr_enc_out, usr_enc_lengths = test_seq, test_seq_lengths
        seq_len = usr_enc_out.shape[1]

        # usr ablation stuff
        if self.lstm_sets_dict['dec_ablation_setting'] == 'only_ling':
            usr_enc_out[:, :, 300:].fill_(0)
        elif self.lstm_sets_dict['dec_ablation_setting'] == 'only_acous':
            usr_enc_out[:, :, :300].fill_(0)
        # elif self.lstm_sets_dict['dec_ablation_setting'] == 'no_context':
        #   h_inf.zero_()
        #   c_inf.zero_()

        # if (not (self.lstm_sets_dict['train_context']) and self.training) or (not(self.lstm_sets_dict['test_context']) and not(self.training)):
        #   h_inf.zero_()
        #   c_inf.zero_()

        # get sys turn encoding
        if self.lstm_sets_dict['response_encoder_hidden_size']:
            if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                pass
            elif self.lstm_sets_dict['encoder_settings']['use_acous']:
                self.response_encoder_lstm_0.flatten_parameters()
                self.response_encoder_lstm_1.flatten_parameters()
                h_enc_seq_acous, (h_enc_0, _) = self.response_encoder_lstm_0(
                    sys_trn_0)
                h_enc_seq_acous, acous_lengths = pad_packed_sequence(
                    h_enc_seq_acous, batch_first=True, total_length=int(max(kwargs['sys_trn_0_lengths']).data.cpu()))
                h_enc_seq_acous = h_enc_seq_acous.transpose(0, 1)
            else:
                self.response_encoder_lstm_1.flatten_parameters()

            if self.response_1_dim:
                h_enc_seq_ling, (h_enc_1, _) = self.response_encoder_lstm_1(
                    sys_trn_1)
                h_enc_seq_ling, ling_lengths = pad_packed_sequence(
                    h_enc_seq_ling, batch_first=True, total_length=int(max(sys_trn_1_lengths).data.cpu()))
                h_enc_seq_ling = h_enc_seq_ling.transpose(0, 1)

            # enc ablation stuff
            if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                pass
            elif self.lstm_sets_dict['enc_ablation_setting'] == 'only_ling':
                h_enc_seq_acous.fill_(0)
            elif self.lstm_sets_dict['enc_ablation_setting'] == 'only_acous':
                h_enc_seq_ling.fill_(0)

            # concatenate start token and acoustic embedding
            if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                pass
            else:
                first_embed = torch.cat([h_enc_seq_ling[0], self.acous_embeds(torch.LongTensor(
                    [self.acous_emb_wait_tok] * h_enc_seq_ling.shape[1]).to(h_enc_seq_ling.device))], 1)
                last_lings = h_enc_seq_ling[ling_lengths -
                                            1, list(range(len(ling_lengths)))]
                last_embed = torch.cat([last_lings, self.acous_embeds(torch.LongTensor(
                    [self.acous_emb_stop_tok] * h_enc_seq_ling.shape[1]).to(last_lings.device))], 1)
            sys_encodings = []
            for b in range(encoding_batch_size):
                if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                    pass
                else:
                    ling_encs = h_enc_seq_ling[1: ling_lengths[b] - 1, b]
                    acous_indices = torch.LongTensor(
                        [wrd_times for wrd_times in word_times_strts[b][1: word_time_lengths[b]-1]])
                    acous_encs = h_enc_seq_acous[acous_indices, b]
                    sys_enc = torch.cat([ling_encs, acous_encs], -1)
                    sys_enc = torch.cat([first_embed[b].unsqueeze(
                        0), sys_enc, last_embed[b].unsqueeze(0)], 0)
                    sys_encodings.append(sys_enc)

            if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                pass
            elif self.lstm_sets_dict['encoder_settings']['skip_vae']:
                sys_encode_pad = pad_sequence(sys_encodings, batch_first=True)
                self.master_encoder.flatten_parameters()
                sys_enc_packed = pack_padded_sequence(
                    sys_encode_pad, ling_lengths, batch_first=True, enforce_sorted=False)
                sys_master_enc_out, (h_enc_master, _) = self.master_encoder(
                    sys_enc_packed)
                sys_master_unpacked, _ = pad_packed_sequence(
                    sys_master_enc_out, batch_first=True)
                first_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), 0]
                second_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), 1]
                last_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), ling_lengths - 1]
                wait_network_in = torch.cat(
                    [first_master_enc, second_master_enc, last_master_enc], -1)
                wait_network_out = self.wait_network(wait_network_in)
                sys_master_unpacked2 = torch.cat(
                    [wait_network_out.unsqueeze(1), sys_master_unpacked[:, 1:]], 1)
                sys_encodings = [
                    sys_master_unpacked2[b][: ling_lengths[b]]
                    for b in range(encoding_batch_size)
                ]
            else:
                sys_encode_pad = pad_sequence(sys_encodings, batch_first=True)
                self.master_encoder.flatten_parameters()
                sys_enc_packed = pack_padded_sequence(
                    sys_encode_pad, ling_lengths, batch_first=True, enforce_sorted=False)
                sys_master_enc_out, (h_enc_master, _) = self.master_encoder(
                    sys_enc_packed)
                sys_master_unpacked, _ = pad_packed_sequence(
                    sys_master_enc_out, batch_first=True)
                sys_encode_pad = sys_master_unpacked[:, 1:]

                first_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), 0]
                second_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), 1]
                last_master_enc = sys_master_unpacked[range(
                    encoding_batch_size), ling_lengths - 1]
                vae_in = torch.cat(
                    [first_master_enc, second_master_enc, last_master_enc], -1)
                z_vae, mu, log_var = self.VAE(vae_in)
                wait_split, start_split = self.Split_Wait_Start(z_vae)
                first_embed = wait_split
                sys_encodings = [torch.cat([
                    first_embed[b].unsqueeze(0),
                    sys_encode_pad[b][: ling_lengths[b]]
                ], 0) for b in range(encoding_batch_size)]

            # Initialize a bunch of variables
            none_tok = self.hidden_embeds(torch.LongTensor(
                [self.hid_emb_none_tok] * batch_size).to(self.hidden_embeds.weight.device))
            sys_enc_out = none_tok.repeat(1, 1).unsqueeze(0).repeat(
                seq_len, 1, 1)  # first repeat unnecessary?
            attend_current = none_tok.unsqueeze(
                1).repeat(1, max(usr_enc_lengths), 1)
            attend_next = none_tok.unsqueeze(
                1).repeat(1, max(usr_enc_lengths), 1)
            val_tok_curr = (torch.ones(
                [seq_len, batch_size])*self.lstm_sets_dict['wait_tok']).long()
            val_tok_next = (torch.ones(
                [seq_len, batch_size])*self.lstm_sets_dict['wait_tok']).long()
        else:
            loc_lengths = sys_trn_1_lengths.data.cpu().numpy()

            sys_encodings = [
                torch.tensor([1.0] + [0.0]*(loc_lengths[b]-1)
                             ).unsqueeze(1).to(self.inference_lstm.weight_hh_l0.device)
                for b in range(encoding_batch_size)
            ]
        # Perform inference using either autoregression (test) or ground truth (training)
        if self.autoregress:
            samp_outs_NT, prob_outs_NT = [], []
            val_outs = []
            y_trgt_out_NT = y_trgt_dict
            b_info = kwargs['batch_turn_info']
            curr_wrd_indx = [0] * batch_size
            final_hidden_c_inf = [[]] * batch_size
            final_hidden_h_inf = [[]] * batch_size
            if not self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                none_tok = self.hidden_embeds(torch.LongTensor(
                    [self.hid_emb_none_tok]).squeeze().to(self.hidden_embeds.weight.device))
            else:
                none_tok = torch.tensor([0.0]).to(
                    self.inference_lstm.weight_hh_l0.device)

            # get rand_strt_f
            update_strt_fs = [[0] + end_fs[:-1]
                              for end_fs in kwargs['y_end_f'].tolist()]
            rand_strt_f_list, sys_trn_strt_f_list, min_rand_strt, max_rand_strt = [], [], [], []
            for b, (update_strt_f, sys_trn_strt_f, ass_ipu_strt_f, ass_ipu_end_f, ass_usr_trn) in enumerate(
                zip(update_strt_fs, kwargs['y_strt_f'], kwargs['associated_usr_ipu_strt_f'],
                    kwargs['associated_usr_ipu_end_f'], kwargs['ass_usr_turn'])
            ):
                # if there is no maximum amount of time
                if self.lstm_sets_dict['max_rand_strt_test_f'] == -20:
                    rand_strt_f_min = max(
                        [int(update_strt_f[0]), int(ass_ipu_strt_f[0])])
                else:
                    rand_strt_limit = int(
                        sys_trn_strt_f[0]) - self.lstm_sets_dict['max_rand_strt_test_f']
                    rand_strt_f_min = max(
                        [int(update_strt_f[0]), int(ass_ipu_strt_f[0]), rand_strt_limit])

                if not rand_strt_f_min == sys_trn_strt_f[0]:
                    # np.random.seed(kwargs['seed'])
                    rand_strt_f = np.random.randint(
                        rand_strt_f_min, sys_trn_strt_f.data.cpu().numpy()[0])
                else:
                    rand_strt_f = rand_strt_f_min

                rand_strt_f_list.append(rand_strt_f)
                min_rand_strt.append(rand_strt_f_min)
                max_rand_strt.append(sys_trn_strt_f[0])

                if self.lstm_sets_dict['sanity_check_bool']:
                    sys_trn_strt_f_list.append(
                        sys_trn_strt_f.data.cpu().numpy()[0])
            # Process sequence sequentially
            for idx in range(test_seq.shape[1]):
                inf_input_concat_list = []
                inf_input_concat_list.append(usr_enc_out[:, idx].unsqueeze(1))
                sys_enc_out_list, attend_current_test_list, attend_next_test_list = [], [], []
                val_outs_idx = []
                for btch in range(batch_size):

                    if not(self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc') and not(self.lstm_sets_dict['encoder_settings']['skip_vae']) and idx >= rand_strt_f_list[btch] and curr_wrd_indx[btch] == 0:
                        # This is for when we choose between the wait and start tokens

                        sys_enc_out_list.append(
                            sys_encodings[btch][curr_wrd_indx[btch]])
                        attend_current_test_list.append(
                            sys_encodings[btch][curr_wrd_indx[btch]])
                        attend_next_test_list.append(start_split[btch])
                    elif idx >= rand_strt_f_list[btch] and curr_wrd_indx[btch] < int(sys_encodings[btch].shape[0]-1):
                        sys_enc_out_list.append(
                            sys_encodings[btch][curr_wrd_indx[btch]])
                        attend_current_test_list.append(
                            sys_encodings[btch][curr_wrd_indx[btch]])
                        attend_next_test_list.append(
                            sys_encodings[btch][curr_wrd_indx[btch] + 1])
                    elif idx < rand_strt_f_list[btch]:

                        if self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                            sys_enc_out_list.append(none_tok)
                        else:
                            sys_enc_out_list.append(none_tok)

                        if not self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                            attend_current_test_list.append(none_tok)
                            attend_next_test_list.append(none_tok)
                    else:
                        sys_enc_out_list.append(none_tok)

                        if not self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                            attend_current_test_list.append(none_tok)
                            attend_next_test_list.append(none_tok)

                sys_enc_out_test = torch.stack(sys_enc_out_list).unsqueeze(1)
                inf_input_concat_list.append(sys_enc_out_test)
                inf_input = torch.cat(inf_input_concat_list, 2)
                self.inference_lstm.flatten_parameters()
                # inf_out, (h_inf, c_inf) = self.inference_lstm(inf_input, (h_inf, c_inf))
                inf_out, _ = self.inference_lstm(inf_input)
                combined_logits = self.out_dict['NT'](inf_out)
                attn_probs = self.sigmoid(combined_logits)

                y_out['NT'].append(combined_logits)
                logit_NT = y_out['NT'][-1] / self.temperature
                # can also use relaxed bernoulli here
                samp_NT = Bernoulli(logits=logit_NT).sample()
                for btch in range(batch_size):
                    if idx >= rand_strt_f_list[btch] and curr_wrd_indx[btch] < (int(sys_encodings[btch].shape[0])-1):
                        curr_wrd_indx[btch] += int(samp_NT[btch].data.cpu())
                        val_outs_idx.append(torch.LongTensor(
                            [b_info['val_tok_list_sil'][btch][curr_wrd_indx[btch]]]))
                    elif idx < rand_strt_f_list[btch]:
                        val_outs_idx.append(torch.LongTensor([-1]))
                        samp_NT[btch] = torch.LongTensor([-1])
                        attn_probs[btch] = torch.LongTensor([-1])
                    else:
                        val_outs_idx.append(torch.LongTensor([-2]))
                        samp_NT[btch] = torch.LongTensor([-2])
                        attn_probs[btch] = torch.LongTensor([-2])
                    # if curr_wrd_indx[btch] == sys_encodings[btch].shape[0] - 1:
                    #   final_hidden_h_inf[btch] = h_inf[:, btch]
                    #   final_hidden_c_inf[btch] = c_inf[:, btch]
                samp_outs_NT.append(samp_NT)
                prob_outs_NT.append(attn_probs)

            # Timeout stuff
            for btch in range(batch_size):
                # if not len(final_hidden_h_inf[btch]):
                #   # print('not finished sentence')
                #   final_hidden_h_inf[btch] = h_inf[:, btch]
                #   final_hidden_c_inf[btch] = c_inf[:, btch]
                if not curr_wrd_indx[btch]:  # Timeout at end of test seq
                    curr_wrd_indx[btch] = 1

            # h_inf = torch.stack(final_hidden_h_inf, 1)
            # c_inf = torch.stack(final_hidden_c_inf, 1)
            prob_outs_NT = torch.stack(prob_outs_NT, 1)
            samp_outs_NT = torch.stack(samp_outs_NT, 1)
            y_out['NT'] = torch.stack(y_out['NT'], 1).squeeze(2)
            if len(y_out['NT'].shape) == 3:
                y_out['NT'].squeeze_(2)
            if self.lstm_sets_dict['encoder_settings']['skip_vae']:
                sys_enc_out_save, sys_enc_mu, sys_enc_log_var = [], [], []
            else:
                sys_enc_out_save, sys_enc_mu, sys_enc_log_var = z_vae.data, mu.data, log_var.data

        # NO Autoregress
        else:
            # get sys turn encoding
            update_strts_two_sys_turn, sys_trn_end_f_list = [], []
            update_strt_fs = [[0] + end_fs[:-1]
                              for end_fs in kwargs['y_end_f'].tolist()]
            b_info = kwargs['batch_turn_info']
            rand_strt_f_list, sys_trn_strt_f_list, min_rand_strt, max_rand_strt = [], [], [], []
            for enc_btch, (update_strt_f, strt_f, end_f, ass_ipu_strt_f, ass_ipu_end_f, ass_usr_trn) in enumerate(
                zip(update_strt_fs, kwargs['y_strt_f'], kwargs['y_end_f'],
                    kwargs['associated_usr_ipu_strt_f'], kwargs['associated_usr_ipu_end_f'],
                    kwargs['ass_usr_turn'])
            ):
                update_strt_f, sys_trn_strt_f, trn_end_f, ass_ipu_strt_f, ass_ipu_end_f = \
                    update_strt_f[0], strt_f[0], end_f[0], ass_ipu_strt_f[0], ass_ipu_end_f[0]

                if self.lstm_sets_dict['two_sys_turn'] and self.training:
                    loc_btch = enc_btch//2
                    if enc_btch % 2:
                        update_strt_f = update_strt_f + \
                            int(kwargs['y_end_f']
                                [enc_btch-1].data.cpu().numpy())
                        sys_trn_strt_f = strt_f + \
                            kwargs['y_end_f'][enc_btch - 1]
                        trn_end_f = end_f + kwargs['y_end_f'][enc_btch - 1]
                        ass_ipu_strt_f = ass_ipu_strt_f + \
                            kwargs['y_end_f'][enc_btch - 1]
                        ass_ipu_end_f = ass_ipu_end_f + \
                            kwargs['y_end_f'][enc_btch - 1]
                else:
                    loc_btch = enc_btch

                if self.lstm_sets_dict['max_wait_train_f'] == - 20:
                    rand_strt_f_min = max(
                        [int(update_strt_f), int(ass_ipu_strt_f)])
                elif self.lstm_sets_dict['max_wait_train_from_usr_turn'] == True:
                    rand_strt_limit = int(sys_trn_strt_f) - \
                        self.lstm_sets_dict['max_wait_train_f']
                    rand_strt_f_min = max(
                        [int(update_strt_f), rand_strt_limit])
                else:
                    rand_strt_limit = int(sys_trn_strt_f) - \
                        self.lstm_sets_dict['max_wait_train_f']
                    rand_strt_f_min = max(
                        [int(update_strt_f), int(ass_ipu_strt_f), rand_strt_limit])

                testing_rand_bool = self.lstm_sets_dict['full_test_flag'] and not self.training
                if (testing_rand_bool or self.training) and not (rand_strt_f_min == int(sys_trn_strt_f.data)):
                    rand_strt_f = np.random.randint(
                        rand_strt_f_min, int(sys_trn_strt_f))
                else:
                    rand_strt_f = rand_strt_f_min

                if self.lstm_sets_dict['plot_batch'] or testing_rand_bool:
                    min_rand_strt.append(rand_strt_f_min)
                    max_rand_strt.append(sys_trn_strt_f)

                assert rand_strt_f <= sys_trn_strt_f

                if self.lstm_sets_dict['full_over']:
                    y_trgt_dict['NT'][loc_btch,
                                      update_strt_f: rand_strt_f] = -1
                else:
                    if ass_usr_trn == -1:
                        y_trgt_dict['NT'][loc_btch, update_strt_f:] = -1
                    else:
                        y_trgt_dict['NT'][loc_btch,
                                          update_strt_f:rand_strt_f] = -1

                rand_strt_f_list.append(rand_strt_f)
                sys_trn_strt_f_list.append(int(sys_trn_strt_f.data.cpu()))
                sys_trn_end_f_list.append(int(trn_end_f.data.cpu()))
                update_strts_two_sys_turn.append(update_strt_f)
                if not self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                    h_length = sys_enc.shape[1]
                    pos_1_mask = torch.zeros(sys_enc_out.shape[2])
                    pos_1_mask[:h_length] = 1
                    pos_1_mask = pos_1_mask.bool()
                    val_tok_list = b_info['val_tok_list_sil'][enc_btch]
                    if self.lstm_sets_dict['full_test_flag'] and not(self.training):
                        sys_enc_out[rand_strt_f:, loc_btch,
                                    pos_1_mask] = sys_encodings[enc_btch][0]
                    else:
                        sys_enc_out[rand_strt_f: sys_trn_strt_f, loc_btch,
                                    pos_1_mask] = sys_encodings[enc_btch][0]

                    if not(self.lstm_sets_dict['encoder_settings']['skip_vae']):
                        attend_current[loc_btch,
                                       rand_strt_f:sys_trn_strt_f] = wait_split[enc_btch]
                        attend_next[loc_btch,
                                    rand_strt_f:sys_trn_strt_f] = start_split[enc_btch]
                    else:
                        attend_current[loc_btch,
                                       rand_strt_f:sys_trn_strt_f] = sys_encodings[enc_btch][0]
                        attend_next[loc_btch,
                                    rand_strt_f:sys_trn_strt_f] = sys_encodings[enc_btch][1]

                    word_times_length = b_info['word_times_lengths'][enc_btch]
                    word_times_strt_sil = b_info['word_times_strt_list_sil'][enc_btch][1:word_times_length - 1]
                    word_times_end_sil = b_info['word_times_end_list_sil'][enc_btch][1: word_times_length - 1]
                    # For sampling without autoregression during full testing
                    if not (self.lstm_sets_dict['full_test_flag'] and not(self.training)):
                        for wrd_i, (strt_f, end_f) in enumerate(zip(word_times_strt_sil, word_times_end_sil)):
                            strt_f_advanced, end_f_advanced = strt_f + \
                                sys_trn_strt_f, end_f + sys_trn_strt_f
                            wrd_i_advanced = wrd_i + 1
                            sys_enc_out[strt_f_advanced:end_f_advanced, loc_btch,
                                        pos_1_mask] = sys_encodings[enc_btch][wrd_i_advanced]
                            attend_current[loc_btch,
                                           strt_f_advanced: end_f_advanced] = sys_encodings[enc_btch][wrd_i_advanced]
                            attend_next[loc_btch,
                                        strt_f_advanced: end_f_advanced] = sys_encodings[enc_btch][wrd_i_advanced+1]
                            val_tok_curr[strt_f_advanced:end_f_advanced,
                                         loc_btch] = val_tok_list[wrd_i_advanced]
                            val_tok_next[strt_f_advanced:end_f_advanced,
                                         loc_btch] = val_tok_list[wrd_i_advanced + 1]

            prob_outs_NT, samp_outs_NT, val_outs_NT = [], [], []
            sys_enc_out_save, sys_enc_mu, sys_enc_log_var = [], [], []
            y_trgt_out_NT = y_trgt_dict
            inf_input_concat_list = []
            inf_input_concat_list.append(usr_enc_out)

            if self.lstm_sets_dict['response_encoder_hidden_size'] and not self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                sys_enc_out = sys_enc_out.transpose(0, 1)
                inf_input_concat_list.append(sys_enc_out)
            elif self.lstm_sets_dict['enc_ablation_setting'] == 'no_enc':
                sys_enc_out = torch.zeros(list(usr_enc_out.shape[:2]) + [1])
                for b, (r_strt_f, s_trn_strt) in enumerate(zip(rand_strt_f_list, sys_trn_strt_f_list)):
                    b_loc = b // 2
                    sys_enc_out[b_loc, r_strt_f:s_trn_strt] = 1
                inf_input_concat_list.append(sys_enc_out.to(
                    self.inference_lstm.weight_hh_l0.device))

            inf_input = torch.cat(inf_input_concat_list, 2)
            self.inference_lstm.flatten_parameters()
            if self.lstm_sets_dict['full_test_flag'] and not (self.training):
                # inf_out, (_, _) = self.inference_lstm(inf_input, (h_inf, c_inf))
                inf_out, (_, _) = self.inference_lstm(inf_input)
            else:
                inf_input_packed = pack_padded_sequence(
                    inf_input, usr_enc_lengths, batch_first=True, enforce_sorted=False)
                # inf_out_packed, (h_inf, c_inf) = self.inference_lstm(inf_input_packed, (h_inf, c_inf))
                inf_out_packed, _ = self.inference_lstm(inf_input_packed)
                inf_out, _ = pad_packed_sequence(
                    inf_out_packed, batch_first=True)
            y_out['NT'] = self.out_dict['NT'](inf_out).squeeze(2)

            if (not(self.training) and self.lstm_sets_dict['full_test_flag']) or self.lstm_sets_dict['use_fixed_test_prob']:
                logit_NT = y_out['NT'] / self.temperature
                prob_outs_NT = self.sigmoid(logit_NT)
                if self.lstm_sets_dict['use_fixed_test_prob']:
                    prob_outs_NT.fill_(self.lstm_sets_dict['fixed_test_prob'])
                    y_out['NT'].fill_(self.lstm_sets_dict['fixed_test_logit'])
                    samp_outs_NT = Bernoulli(prob_outs_NT).sample()
                else:
                    samp_outs_NT = Bernoulli(logits=logit_NT).sample()
                for bt, strt_f_samp in enumerate(rand_strt_f_list):
                    samp_outs_NT[bt, :strt_f_samp] = 0
                    prob_outs_NT[bt, :strt_f_samp] = 0.0

            if self.training or self.lstm_sets_dict['encoder_settings']['skip_vae']:
                sys_enc_out_save, sys_enc_mu, sys_enc_log_var = [], [], []
            else:
                sys_enc_out_save, sys_enc_mu, sys_enc_log_var = z_vae.data, mu.data, log_var.data

        # Loss calculation
        mod_out = {
            'NT': y_out['NT'],
        }
        loss = torch.tensor(0.0).to(self.inference_lstm.weight_hh_l0.device)
        bp_loss = torch.tensor(0.0).to(self.inference_lstm.weight_hh_l0.device)
        num_pred_samples_for_batch = {
            task: 0 for task in self.pred_task_dict['active_outputs']}
        num_pred_samples_for_result = {
            task: 0 for task in self.pred_task_dict['active_outputs']}
        loss_dict_train_raw = {
            task: 0.0 for task in self.pred_task_dict['active_outputs']}
        loss_dict_train_raw['all'] = 0.0

        for task in self.pred_task_dict['active_outputs']:

            if task == 'KLD':
                mask = torch.zeros(log_var.shape, dtype=torch.bool)
                for b in range(encoding_batch_size):
                    mask[b] = 1 if (kwargs['ass_usr_turn'][b] != -1) else 0
                n_samps = torch.sum(mask)
                if n_samps:
                    # don't need (1.0/self.lstm_sets_dict['vae_dim'])
                    loss_task = -0.5 * \
                        torch.sum(
                            1 + log_var[mask] - mu[mask].pow(2) - log_var[mask].exp())
                    num_pred_samples_for_batch[task] = torch.sum(mask)
                else:
                    loss_task = torch.tensor(0.0).to(
                        self.inference_lstm.weight_hh_l0.device)
                    loss_task.requires_grad = True
                    num_pred_samples_for_batch[task] += 1

            elif task == 'TL':
                y_out_loc = mod_out['NT']
                seq_len = y_out_loc.shape[1]
                y_trgt_loc = y_trgt_dict['NT'][:, :seq_len].clone().to(
                    self.inference_lstm.weight_hh_l0.device)
                mask = torch.zeros(y_trgt_loc.shape, dtype=torch.bool)
                y_strt_f = sys_trn_strt_f_list
                for b, (r_strt, sys_strt) in enumerate(zip(rand_strt_f_list, y_strt_f)):
                    if self.lstm_sets_dict['two_sys_turn'] and self.training:
                        b = b//2
                    mask[b, r_strt:sys_strt] = 1
                trgt_TL = torch.ones(mask.shape).to(
                    self.inference_lstm.weight_hh_l0.device) * -1
                trgt_TL[mask] = y_trgt_loc[mask]
                if self.lstm_sets_dict['sanity_check_bool'] or self.lstm_sets_dict['plot_batch']:
                    y_trgt_out_NT['TL'] = trgt_TL
                n_samps = np.sum(mask.cpu().numpy())

                if n_samps:
                    loss_task = self.loss_func_BCE_logit(
                        y_out_loc[mask], y_trgt_loc[mask])
                    num_pred_samples_for_batch[task] += n_samps
                else:
                    loss_task = torch.tensor(0.0).to(
                        self.inference_lstm.weight_hh_l0.device)
                    loss_task.requires_grad = True
                    num_pred_samples_for_batch[task] += 1

                if self.lstm_sets_dict['analyze_error']:
                    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                    loss_analysis = criterion(
                        y_out_loc[mask], y_trgt_loc[mask])
                    analysis_tmp = torch.zeros(mask.shape).cuda()
                    analysis_tmp[mask] = loss_analysis / n_samps
                    analysis_loss_per_batch = torch.sum(analysis_tmp, 1)
                    analysis_count = torch.sum(mask, 1)
                else:
                    analysis_count = []
                    analysis_loss_per_batch = []

            else:  # for 'NT'
                y_out_loc = mod_out[task]
                seq_len = y_out_loc.shape[1]
                y_trgt_loc = y_trgt_dict[task][:, :seq_len].clone().to(
                    self.inference_lstm.weight_hh_l0.device)
                mask = y_trgt_loc != -1.0
                n_samps = np.sum(mask.cpu().numpy())
                if n_samps:
                    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
                    loss_task = criterion(y_out_loc[mask], y_trgt_loc[mask])
                    num_pred_samples_for_batch[task] += n_samps
                    # comment start for batch_plot
                    if self.lstm_sets_dict['plot_batch'] or self.lstm_sets_dict['sanity_check_bool']:
                        prob_outs_NT = self.sigmoid(y_out_loc)
                        trgt_NT = torch.ones(mask.shape).cuda() * -1
                        trgt_NT[mask] = y_trgt_loc[mask]
                        if self.lstm_sets_dict['sanity_check_bool'] or self.lstm_sets_dict['plot_batch']:
                            y_trgt_out_NT['NT'] = trgt_NT
                else:
                    loss_task = torch.tensor(0.0).to(
                        self.inference_lstm.weight_hh_l0.device)
                    loss_task.requires_grad = True
                    num_pred_samples_for_batch[task] += 1

            num_pred_samples_for_result[task] += num_pred_samples_for_batch[task]
            loss_dict_train_raw[task] += loss_task.item()
            bp_loss += self.pred_task_dict[task]['weight'] * \
                loss_task / num_pred_samples_for_batch[task]
        if cuda_avail:
            num_pred_samples_for_result = {k: torch.LongTensor(
                [v]).cuda() for k, v in num_pred_samples_for_result.items()}
            num_pred_samples_for_batch = {k: torch.LongTensor(
                [v]).cuda() for k, v in num_pred_samples_for_batch.items()}
            loss_dict_train_raw = {k: torch.FloatTensor(
                [v]).cuda() for k, v in loss_dict_train_raw.items()}
        else:
            num_pred_samples_for_result = {k: torch.LongTensor(
                [v]) for k, v in num_pred_samples_for_result.items()}
            num_pred_samples_for_batch = {k: torch.LongTensor(
                [v]) for k, v in num_pred_samples_for_batch.items()}
            loss_dict_train_raw = {k: torch.FloatTensor(
                [v]) for k, v in loss_dict_train_raw.items()}

        if self.lstm_sets_dict['plot_batch'] or self.lstm_sets_dict['sanity_check_bool'] or self.lstm_sets_dict['full_test_flag'] or self.lstm_sets_dict['test_autoregress']:
            rand_strt_dict = {
                'min_rand_strt_f': torch.tensor(min_rand_strt),
                'max_rand_strt_f': torch.tensor(max_rand_strt),
                'sampled_rand_strt': torch.tensor(rand_strt_f_list)
            }
        else:
            rand_strt_dict = []

        outputs = {
            # 'h_inf': h_inf.squeeze(0),
            # 'c_inf': c_inf.squeeze(0),
            'num_pred_samples_for_result': num_pred_samples_for_result,
            'num_pred_samples_for_batch': num_pred_samples_for_batch,
            'loss_dict_train_raw': loss_dict_train_raw,
            'prob_outs': prob_outs_NT,
            'samp_outs': samp_outs_NT,
            'y_trgt_out': y_trgt_out_NT,
            'sys_enc_out_save': sys_enc_out_save,
            'sys_enc_mu': sys_enc_mu,
            'sys_enc_log_var': sys_enc_log_var,
            'analysis_loss_per_batch': analysis_loss_per_batch,
            'analysis_count': analysis_count,
            'rand_strt_dict': rand_strt_dict
        }
        return torch.unsqueeze(bp_loss, 0), outputs


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim1, z_dim)
        self.fc32 = nn.Linear(h_dim1, z_dim)
        self.stat_bool = False

    def encoder(self, x):
        x1 = F.relu(self.fc1(x))
        # x2 = F.relu(self.fc2(x1))
        return self.fc31(x1), self.fc32(x1)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def static_sampling(self, mu, log_var):
        std = torch.exp(
            0.5*self.stat_log_var.unsqueeze(0).expand(mu.shape[0], -1))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(self.stat_mu)  # return z sample

    def set_static_mu_log_var(self, stat_mu, stat_log_var):
        self.stat_bool = True
        self.stat_mu = torch.tensor(stat_mu).to(self.fc1.weight.device)
        self.stat_log_var = torch.tensor(
            stat_log_var).to(self.fc1.weight.device)
        self.sampling = self.static_sampling

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return z, mu, log_var


class Split_Wait_Start(nn.Module):
    def __init__(self, vae_dim, enc_reduce_size):
        super(Split_Wait_Start, self).__init__()
        self.fc1 = nn.Linear(vae_dim, enc_reduce_size)
        self.fc21 = nn.Linear(enc_reduce_size, enc_reduce_size)
        self.fc22 = nn.Linear(enc_reduce_size, enc_reduce_size)
        self.enc_reduce_size = enc_reduce_size

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        wait_split = F.relu(self.fc21(x1))
        start_split = F.relu(self.fc22(x1))
        return wait_split, start_split


class Reduce_Sys_Enc(nn.Module):
    def __init__(self, sys_enc_size, enc_reduce_size):
        super(Reduce_Sys_Enc, self).__init__()
        self.fc1 = nn.Linear(sys_enc_size, enc_reduce_size)

    def forward(self, x):
        return F.relu(self.fc1(x))
