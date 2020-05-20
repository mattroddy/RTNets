#!/usr/bin/env python3 -u
"""
Main Script for training and testing
"""
import argparse
import json
import logging
import os
import pdb
import random
import sys
import time as t
from collections import OrderedDict

import numpy as np
import spacy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

from data_loader import ContDataset, MySampler
from dataset_settings import data_dir
# from extra import plot_batch
from model import ContModel
from settings import (args, batch_size, device,
                      individual_turnpair_experiments, just_test,
                      just_test_folder, just_test_model, language_size,
                      load_encodings, load_model, load_second_encodings,
                      lstm_sets_dict, max_epochs, naming_dict, note_append,
                      num_data_loader_workers)
from settings import num_feat_per_person as num_feat_per_person_dict
from settings import (optim_patience, pred_task_dict,
                      test_dataset_settings_dict, test_file_list,
                      time_out_length, train_dataset_settings_dict,
                      train_file_list, use_ling, vae_data_multiplier,
                      vae_data_multiplier_2, vae_experiments, vae_target_da,
                      valid_dataset_settings_dict, valid_file_list)
from test_funcs import test
from util import (get_individual_turnpair_dataset, get_vae_dataset,
                  get_vae_encodings)

# from test_funcs import sanity_check_func, get_batch_items_for_full_test

sys.dont_write_bytecode = True
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

SET_TRAIN = 0
SET_VALID = 1
SET_TEST = 2


def main():

    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    t0 = t.time()
    embeds_usr, embeds_sys, embeds, num_feat_per_person, sil_tok, nlp = setup_ling()

    if just_test:
        just_test_func(
            embeds_usr=embeds_usr,
            embeds_sys=embeds_sys,
            embeds=embeds,
            num_feat_per_person=num_feat_per_person,
            sil_tok=sil_tok,
            nlp=nlp
        )

    def _init_fn(): return np.random.seed(SEED)

    print('Loading valid DATA')
    valid_dataset = ContDataset(valid_dataset_settings_dict)
    valid_dataset.embeds_usr = embeds_usr
    valid_dataset.embeds_sys = embeds_sys
    valid_dataset.nlp = nlp
    valid_dataset.sil_tok = sil_tok
    collate_fn_valid = valid_dataset.collate_fn
    valid_sampler = MySampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  batch_size=batch_size, collate_fn=collate_fn_valid,
                                  num_workers=num_data_loader_workers,
                                  worker_init_fn=_init_fn)
    num_valid_batches = len(valid_dataloader)
    valid_dataset.update_annots_test = test_dataset_settings_dict['update_annots_test']
    print('Loading train DATA')
    train_dataset = ContDataset(train_dataset_settings_dict)
    train_dataset.embeds_usr = embeds_usr
    train_dataset.embeds_sys = embeds_sys
    train_dataset.nlp = nlp
    train_dataset.sil_tok = sil_tok
    collate_fn_train = train_dataset.collate_fn
    if lstm_sets_dict['two_sys_turn']:
        tmp_sampler = MySampler(train_dataset)
        train_sampler = SubsetRandomSampler(tmp_sampler.my_indices_no_first)
    else:
        train_sampler = RandomSampler(
            train_dataset) if lstm_sets_dict['train_random_sample'] else MySampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=batch_size, collate_fn=collate_fn_train,
                                  num_workers=num_data_loader_workers, worker_init_fn=_init_fn)
    num_train_batches = len(train_dataloader)
    print('Done loading all DATA')
    print('time taken: \n' + str(t.time() - t0))
    context_vec_settings_dict = {
        'train': len(train_file_list),
        'valid': len(valid_file_list),
        'test': len(test_file_list)
    }
    lstm_sets_dict['sil_tok'] = sil_tok
    model = ContModel(num_feat_per_person, lstm_sets_dict,
                      device, context_vec_settings_dict,
                      embeds_usr, embeds_sys, embeds)
    # print([[k, i.shape, torch.prod(torch.tensor(i.shape))] for k, i in model.state_dict().items()]) # keep for debugging
    # best_valid_loss = 10000
    iteration = 0
    results_dict = OrderedDict()
    results_dict['train'], results_dict['valid'], results_dict['test'] = OrderedDict(
    ), OrderedDict(), OrderedDict()
    for task in ['all'] + pred_task_dict['active_outputs'] + ['iteration', 'epoch', ]:
        results_dict['train'][task] = []
        results_dict['valid'][task] = []
        results_dict['test'][task] = []
    # results_dict['test']['num_batches'] = len(test_dataloader)
    results_dict['valid']['stats'] = []
    results_dict['test']['stats'] = []

    if load_model:
        print('LOADING MODEL FROM DISK')
        if torch.cuda.is_available():
            checkpoint = torch.load(just_test_folder + '/model.pt')
        else:
            checkpoint = torch.load(
                just_test_folder+'/model.pt', map_location='cpu')
        model = torch.nn.DataParallel(model, dim=0)
        model.load_state_dict(checkpoint)
        model.to(device)
        embeds = model.module.embeds
        train_dataset.embeds = embeds
        valid_dataset.embeds = embeds
        train_dataset.nlp = nlp
        valid_dataset.nlp = nlp
        valid_dataset.sil_tok = sil_tok

        results_dict = json.load(open(just_test_folder + '/results.json', 'r'))
        # model.load_state_dict(checkpoint, strict=False)
        iteration = results_dict['train']['iteration'][-1]
        if not note_append == '_dev' and not os.path.exists(just_test_folder+'/optimizer.pt'):
            initial_learning_rate = float(input("Set initial learning rate:"))
            lstm_sets_dict['learning_rate'] = initial_learning_rate
    else:
        model = torch.nn.DataParallel(model, dim=0)
        model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    print('Parameter count:{}'.format(int(count_parameters(model))))
    optimizer = torch.optim.Adam(model.parameters(
    ), lr=lstm_sets_dict['learning_rate'], weight_decay=lstm_sets_dict['l2'])
    if load_model and os.path.exists(just_test_folder + '/optimizer.pt'):
        optim_state = torch.load(just_test_folder+'/optimizer.pt')
        optimizer.load_state_dict(optim_state)
        print('optimizer loaded. LR:{}'.format(optimizer.defaults['lr']))

    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=optim_patience, min_lr=5.0e-06, verbose=True)
    # 9000, 2000, 2000, 1000, 1000 iterations.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lstm_sets_dict['milestones'], gamma=0.1)
    if load_model and os.path.exists(just_test_folder + '/scheduler.pt'):
        sched_state = torch.load(just_test_folder + '/scheduler.pt')
        scheduler.load_state_dict(sched_state)
        print('scheduler loaded.')

    print('LR {}'.format(get_lr(optimizer)))

    # test_dataset.embeds = model.module.embeds
    train_dataset.embeds = model.module.embeds
    valid_dataset.embeds = model.module.embeds

    # Train
    for epoch in range(max_epochs):
        model.train()
        loss_dict_train_raw = {
            task: 0.0 for task in pred_task_dict['active_outputs']}
        loss_dict_train_raw['all'] = 0.0
        num_pred_samples_for_result = {
            task: 0 for task in pred_task_dict['active_outputs']}
        model.module.reset_hidden('train')
        # hidden_inference = model.module.hidden_inference['train']
        model.zero_grad()
        start_time = t.time()
        for batch_ndx, batch in enumerate(train_dataloader):
            if not (lstm_sets_dict['two_sys_turn']) and (len(batch['update_strt_f']) != batch_size):
                # This should just be triggered for the last few batches of the epoch
                continue

            if lstm_sets_dict['two_sys_turn'] and batch['sys_trn_1'].shape[0] != int(batch_size * 2):
                print('caught small batch')
                continue

            cont_file_indx, cont_ab_indx = batch['file_idx'], batch['a_idx']
            if lstm_sets_dict['two_sys_turn']:
                cont_file_indx = cont_file_indx[::2]
                cont_ab_indx = cont_ab_indx[::2]
            # h_inf = hidden_inference[:, cont_file_indx, cont_ab_indx, 0, :]
            # c_inf = hidden_inference[:, cont_file_indx, cont_ab_indx, 1, :]
            mod_in = {k: v for k, v in batch.items() if not (k in ['y_dict'])}
            # mod_in['h_inf'] = h_inf.squeeze(0)
            # mod_in['c_inf'] = c_inf.squeeze(0)
            mod_in = {**batch['y_dict'], **mod_in}
            # loc_seed = torch.LongTensor([random.randint(0, 1<<31)]*2).unsqueeze(1)
            # mod_in['seed'] = loc_seed
            bp_loss, outputs = model(**mod_in)
            loss = torch.sum(bp_loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if lstm_sets_dict['plot_batch']:
            #     for b_i in [0, 1, 2, 3]:
            #         plot_batch(outputs, mod_in, train_dataset, b_i)
            #     quit()
            # Don't delete, need for context
            # hidden_inference[:, cont_file_indx, cont_ab_indx, 0, :] = outputs['h_inf'].detach().cpu()
            # hidden_inference[:, cont_file_indx, cont_ab_indx, 1, :] = outputs['c_inf'].detach().cpu()

            # aggregate info
            loss_dict_train_raw = {k: float(loss_dict_train_raw[k]) + float(np.sum(
                v.data.cpu().numpy())) for k, v in outputs['loss_dict_train_raw'].items()}
            num_pred_samples_for_result = {k: num_pred_samples_for_result[k] + int(np.sum(
                v.data.cpu().numpy())) for k, v in outputs['num_pred_samples_for_result'].items()}

            if (iteration + 1) % 10 == 0 or ((note_append == '_dev' or note_append == '_dev_restart_') and (iteration + 1) % 2 == 0):
                print_results = {}
                print_results['all'] = 0.0
                weight_denom = 0.0
                for task in pred_task_dict['active_outputs']:
                    print_results[task] = loss_dict_train_raw[task] / \
                        num_pred_samples_for_result[task]
                    print_results['all'] += pred_task_dict[task]['weight'] * \
                        print_results[task]
                    weight_denom += pred_task_dict[task]['weight']
                    num_pred_samples_for_result[task] = 0
                    loss_dict_train_raw[task] = 0.0
                print_results['all'] = print_results['all']/weight_denom

                elapsed = t.time() - start_time
                loss_string = ''
                loss_string += ' train | epoch {:2d} {:4d}/{:4d}| dur(s) {:4.2f} |'
                loss_string += ''.join(
                    [task + ' {:1.5f} |' for task in pred_task_dict['active_outputs']])
                loss_string += ' Weighted {:1.5f} '
                loss_string_items = [epoch, batch_ndx+1, num_train_batches, elapsed] + [
                    print_results[task] for task in pred_task_dict['active_outputs']] + [print_results['all']]
                print(loss_string.format(*loss_string_items))
                for task in pred_task_dict['active_outputs']:
                    results_dict['train'][task].append(
                        float(print_results[task]))
                results_dict['train']['all'].append(
                    float(print_results['all']))
                results_dict['train']['iteration'].append(int(iteration) + 1)
                results_dict['train']['epoch'].append(int(epoch))
                start_time = t.time()
            if (iteration + 1) % 200 == 0 or ((note_append == '_dev' or note_append == '_dev_restart_') and (iteration + 1) % 2 == 0):  # 25
                full_test_flag = lstm_sets_dict['valid_full_test_flag']
                model.module.autoregress = lstm_sets_dict['valid_autoregress']
                valid_loss_all, valid_loss_TL = test(
                    model, valid_dataloader, full_test_flag, results_dict, iteration, epoch)
                if note_append != '_dev' and (np.argmin(valid_loss_TL) == (len(valid_loss_TL)-1)):
                    torch.save(model.state_dict(),
                               naming_dict['fold_name']+'/best_model.pt')
                torch.save(model.state_dict(),
                           naming_dict['fold_name'] + '/model.pt')
                torch.save(optimizer.state_dict(),
                           naming_dict['fold_name']+'/optimizer.pt')
                json.dump(results_dict, open(
                    naming_dict['fold_name'] + '/results.json', 'w'), indent=4)
                # scheduler.step(valid_loss_all[-1])
                scheduler.step()
                torch.save(scheduler.state_dict(),
                           naming_dict['fold_name']+'/scheduler.pt')
                print(naming_dict['fold_name'])
                print('LR {}'.format(get_lr(optimizer)))
                print('Best TL valid loss: {:.4f} ({} steps ago) \n'.format(
                    np.min(valid_loss_TL), len(valid_loss_TL) - np.argmin(valid_loss_TL)))
                # Run tests after final iteration
                if scheduler._step_count >= scheduler.milestones[-1]:
                    # load test dataloader
                    del train_dataset.dataset  # free some RAM
                    test_dataset = ContDataset(test_dataset_settings_dict)
                    collate_fn_test = test_dataset.collate_fn
                    test_dataset.embeds_usr = embeds_usr
                    test_dataset.embeds_sys = embeds_sys
                    test_dataset.embeds = embeds
                    test_dataset.nlp = nlp
                    test_dataset.sil_tok = sil_tok
                    test_sampler = MySampler(test_dataset)
                    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                                 batch_size=batch_size, collate_fn=collate_fn_test,
                                                 num_workers=num_data_loader_workers,
                                                 worker_init_fn=_init_fn)
                    test_dataset.update_annots_test = test_dataset_settings_dict['update_annots_test']
                    test_dataloader.dataset.time_out_length = time_out_length
                    epoch = 0
                    train_batch_indx = -1
                    full_test_flag = False
                    test(model, test_dataloader, full_test_flag,
                         results_dict, train_batch_indx, epoch)
                    json.dump(results_dict, open(
                        naming_dict['fold_name'] + '/results_test.json', 'w'), indent=4)
                    print('Finished non-sampling test')
                    full_test_flag = True
                    model.module.lstm_sets_dict['full_test_flag'] = True
                    test(model, test_dataloader, full_test_flag,
                         results_dict, train_batch_indx, epoch)
                    json.dump(results_dict, open(
                        naming_dict['fold_name'] + '/results_sampled.json', 'w'), indent=4)
                    print('Finished sampling test')
                    print('DONE')
                    os._exit(0)
                model.train()
                model.module.autoregress = lstm_sets_dict['train_autoregress']
                start_time = t.time()
            iteration += 1
        start_time = t.time()
    print('finished')


def setup_ling():
    if use_ling:
        nlp = spacy.blank('en')
        if language_size == 500:
            print('using REALLY small language: 500')
            nlp.from_disk(data_dir+'/spacy_tok_combined_500/')
        elif language_size == 5000:
            print('using small language: 5000')
            nlp.from_disk(data_dir+'/spacy_tok_combined_5000/')
        elif language_size == 10000:
            print('using small language: 10000')
            nlp.from_disk(data_dir+'/spacy_tok_combined_10000/')
        else:
            print('using medium language:20000')
            nlp.from_disk(data_dir+'/spacy_tok_combined_20000/')
        spacy.vocab.link_vectors_to_models(nlp.vocab)
        unspec_tok = len(nlp.vocab.vectors.data)
        sil_tok = unspec_tok + 1
        if lstm_sets_dict['use_wait_stop_tok']:
            lstm_sets_dict['unspec_tok'] = unspec_tok  # for user
            lstm_sets_dict['sil_tok'] = sil_tok
            lstm_sets_dict['wait_tok'] = sil_tok + 1
            lstm_sets_dict['stop_tok'] = sil_tok + 2
            lstm_sets_dict['pad_tok'] = sil_tok + 3
            num_embed_rows_to_add = 5
            # padding_idx = lstm_sets_dict['stop_tok']
            padding_idx = lstm_sets_dict['pad_tok']
        else:
            num_embed_rows_to_add = 1
            # padding_idx = sil_tok
            padding_idx = None
        lstm_sets_dict['sil_tok'] = sil_tok
        embedding_dim = nlp.vocab.vectors.data.shape[1]
        num_embeddings = len(nlp.vocab.vectors.data)
        if lstm_sets_dict['ling_use_glove']:
            embeds = nn.Embedding.from_pretrained(
                torch.FloatTensor(np.concatenate([np.array(nlp.vocab.vectors.data), np.zeros(
                    [num_embed_rows_to_add, embedding_dim])])),
                padding_idx=padding_idx, freeze=lstm_sets_dict['ling_emb_freeze']
            )
        else:
            num_embeddings = len(nlp.vocab.vectors.data)
            embeds = nn.Embedding(
                num_embeddings + 1, embedding_dim=embedding_dim, padding_idx=sil_tok).to(device)

        embeds_reduce_layer_usr = nn.Linear(embedding_dim, 300)
        embeds_reduce_layer_sys = nn.Linear(embedding_dim, 300)
        embeds_dropout_usr = nn.Dropout(lstm_sets_dict['embeds_dropout'])
        embeds_dropout_sys = nn.Dropout(lstm_sets_dict['embeds_dropout'])
        embeds_usr = nn.Sequential(embeds_dropout_usr, embeds_reduce_layer_usr)
        embeds_sys = nn.Sequential(embeds_dropout_sys, embeds_reduce_layer_sys)
        num_feat_per_person = num_feat_per_person_dict['acous'] + embedding_dim
        print('Embeddings loaded.')
    else:
        num_feat_per_person = num_feat_per_person_dict['acous']
        embeds_usr, embeds_sys = 0, 0
        sil_tok = -1
        nlp = -1
    return embeds_usr, embeds_sys, embeds, num_feat_per_person, sil_tok, nlp


def just_test_func(**kwargs):
    print('******* JUST TESTING *****')
    print('Loading test DATA')
    context_vec_settings_dict = {
        'train': len(train_file_list),
        'valid': len(valid_file_list),
        'test': len(test_file_list)
    }
    # if kwargs['load_test_model']:
    if torch.cuda.is_available():
        checkpoint = torch.load(just_test_folder+just_test_model)
    else:
        checkpoint = torch.load(
            just_test_folder+just_test_model, map_location='cpu')
    model = ContModel(kwargs['num_feat_per_person'], lstm_sets_dict, device,
                      context_vec_settings_dict, kwargs['embeds_usr'], kwargs['embeds_sys'], kwargs['embeds'])
    model.temperature = lstm_sets_dict['temperature']
    model.autoregress = lstm_sets_dict['test_autoregress']

    # only test on one gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0], dim=0)
    else:
        model = torch.nn.DataParallel(model, dim=0)
    strict = True
    model.load_state_dict(checkpoint, strict=strict)

    model.to(device)
    model.eval()
    embeds = model.module.embeds

    test_dataset = ContDataset(test_dataset_settings_dict)
    collate_fn_test = test_dataset.collate_fn
    test_dataset.embeds_usr = kwargs['embeds_usr']
    test_dataset.embeds_sys = kwargs['embeds_sys']
    test_dataset.embeds = embeds
    test_dataset.nlp = kwargs['nlp']
    test_dataset.sil_tok = kwargs['sil_tok']
    def _init_fn(): return np.random.seed(lstm_sets_dict['seed'])
    if vae_experiments:
        test_dataset_subset = get_vae_dataset(
            test_dataset, test_dataset_settings_dict['update_annots_test'], vae_target_da)
        test_dataset_subset = test_dataset_subset * vae_data_multiplier
        if load_encodings and load_second_encodings:
            test_dataset_subset_2 = get_vae_dataset(
                test_dataset, test_dataset_settings_dict['update_annots_test'], vae_target_da)
            test_dataset_subset_2 = test_dataset_subset * vae_data_multiplier_2
            test_dataset_subset = test_dataset_subset[:len(
                test_dataset_subset)//2] + test_dataset_subset_2[:len(test_dataset_subset_2)//2]
        print('Target da: {}\t number of points: {}'.format(
            vae_target_da, len(test_dataset_subset)))
        test_sampler = SubsetRandomSampler(test_dataset_subset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                     batch_size=batch_size, collate_fn=collate_fn_test,
                                     num_workers=num_data_loader_workers, drop_last=False,
                                     worker_init_fn=_init_fn)
        if load_encodings:
            mu, log_var = get_vae_encodings(lstm_sets_dict, False)
            if load_second_encodings:
                mu_2, log_var_2 = get_vae_encodings(lstm_sets_dict, True)
                mu = (mu + mu_2) / 2
                log_var = np.log(0.5*(np.exp(log_var)+np.exp(log_var_2)))
            model.module.VAE.set_static_mu_log_var(mu, log_var)
    elif individual_turnpair_experiments:
        test_dataset_subset = get_individual_turnpair_dataset(test_dataset, test_dataset_settings_dict['update_annots_test'],
                                                              target_individual_turnpairs[0], target_individual_turnpairs[1], target_individual_turnpairs[2])
        test_dataset_subset = test_dataset_subset * vae_data_multiplier
        print('Target da: {}\t number of points: {}'.format(
            vae_target_da, len(test_dataset_subset)))
        test_sampler = SubsetRandomSampler(test_dataset_subset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                     batch_size=batch_size, collate_fn=collate_fn_test,
                                     num_workers=num_data_loader_workers, drop_last=False,
                                     worker_init_fn=_init_fn)
    else:
        test_sampler = MySampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                     batch_size=batch_size, collate_fn=collate_fn_test,
                                     num_workers=num_data_loader_workers,
                                     worker_init_fn=_init_fn)

    test_dataset.update_annots_test = test_dataset_settings_dict['update_annots_test']

    print('Testing with temperature:' + str(lstm_sets_dict['temperature']))
    # print('Number of Test files: '+ str(test_dataset.file_list))
    test_dataloader.dataset.time_out_length = time_out_length
    epoch = 0
    # print('Just testing...')
    results_dict = json.load(open(just_test_folder + '/results.json', 'r'))
    train_batch_indx = -1
    full_test_flag = True if lstm_sets_dict['full_test_flag'] else lstm_sets_dict['test_autoregress']
    # pdb.set_trace()
    test(model, test_dataloader, full_test_flag,
         results_dict, train_batch_indx, epoch)
    json.dump(results_dict, open(
        naming_dict['fold_name'] + '/results.json', 'w'), indent=4)
    os._exit(0)


if __name__ == '__main__':
    main()
