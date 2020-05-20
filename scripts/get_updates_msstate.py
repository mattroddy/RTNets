# import argparse
import multiprocessing
import os
import pickle
import sys
import time as t
# import xml.etree.ElementTree
from copy import deepcopy

import nltk
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Dialogue act tagging imports
# from DialogueAct_Tagger_master.config import Config, Model
# from DialogueAct_Tagger_master.predictors.my_predictor import MYPredictor
# # import argparse
# from DialogueAct_Tagger_master.predictors.svm_predictor import SVMPredictor
from get_updates import (convert_to_ms_int, deal_with_disfluency, get_turns,
                         get_updates_func)

# # Notes on DAs
# """
# The previous "sentence" is actually the previous dialog act that was
# produced by the SAME person. Their github code is misleading.
# """

# parser = argparse.ArgumentParser(
#     description='DialogueActTag - Tag a sentence with the ISO dialogue act taxonomy')
# parser.add_argument('-model', dest='model', type=str,
#                     help='the model folder to use for prediction')
# parser.add_argument('-class', dest='layer', type=str, default="all",
#                     help='which level of the taxonomy to tag. Options are: \n'
#                     'all -- trains all the classifiers (default)'
#                     'dim -- trains the dimension classifier'
#                     'task -- trains the task CF classifier'
#                     'som -- trains the SOM CF classifier')
# parser.add_argument('-s', dest='sentence', type=str,
#                     help="the sentence to tag")
# parser.add_argument('-p', dest='prev', type=str,
#                     help="[optional] the previous sentence in the dialogue")

# args = parser.parse_args()
# # if args.model is None or args.sentence is None:
# #   parser.print_help(sys.stderr)
# #   exit(1)
# args.model = '/run/media/matt/Data/Dropbox/SWB_Data/DialogueAct_Tagger_master/models/Model.SVM/'
# args.sentence = 'what am i'
# # logger.info("Restoring model config from meta.json")
# cfg = Config.from_json(f"{args.model}/meta.json")
# if cfg.model_type == Model.SVM:
#     # logger.info("Loading SVM tagger")
#     print("Loading SVM tagger")
#     # tagger = SVMPredictor(cfg)
#     tagger = MYPredictor(cfg)
#     tag_classes = tagger.damsl_model.classes_
# else:
#     raise NotImplementedError(f"Unknown classifier type: {cfg.model_type}")
# # logger.info("Tagging utterance")
# sentences = [['i am testing this', 'what are you doing'],
#              ['yes', 'do you agree']]
# # prev_sentences = [['what are you doing'],['do you agree']]
# print(tagger.dialogue_act_tag_multi(sentences))


# takes 1 min with 8 cores
num_cores = 8
ipu_thresh = 0.200  # 200ms threshold for ipu segmentation
frame_length = 0.050  # 50ms frame size
# ling_feat_delay = 0.100 # 100ms word delay to simulate ASR

if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0  # 0 for 50ms, 1 for 10ms


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


t_1 = t.time()

data_dir = os.popen('sh ./paths.sh').read().rstrip()
audio_files_dir = data_dir + '/mono_dir/'
path_to_features = data_dir + '/gemaps_features_processed_50ms/znormalized/'
path_to_annotations = data_dir + '/swb_ms98_transcriptions/'
path_to_output_trans_files = data_dir + '/ms98_processed_txt/'

# files_annotation_list = list(pd.read_csv(
#     './signals/complete_ms_state.txt', header=None, dtype=str)[0])

files_annotation_list = list(pd.read_csv(
    './splits/complete_ms_state.txt',
    header=None, dtype=str)[0])


# files with aside
# files_annotation_list = [
#     'sw2867',
#     'sw4905',
#     'sw3357'
#                          ]
# files_annotation_list = [
#   'sw2121'
# ]
# A
files_feature_list, files_output_list = [], []
files_terms, files_DA = [], []
files_feature_list, files_output_list = [], []
files_terms, files_DA = [], []


# %% First get vocabulary
no_change, added_to_end_list, max_len = 0, 0, 0
lengths_list, longer_than_one_list = [], []
# word_to_ix = pickle.load(open('./extracted_annotations/word_to_ix.p', 'rb'))


def get_ipus_turns(a_b, data_dict, get_turns_input_list):
    turn_ipu_list_A, full_overlap_list_A, prev_gap_silence_bools_A = get_turns(
        get_turns_input_list)
    turn_ipu_start_indx_A = np.where(
        np.array([-1] + turn_ipu_list_A) != np.array(turn_ipu_list_A + [turn_ipu_list_A[-1]]))[0]
    turn_words_start_indx_A = data_dict[a_b]['ipu_start_indices'][turn_ipu_start_indx_A]
    turn_words_end_indx_A = np.append(
        data_dict[a_b]['ipu_end_indices'][(turn_ipu_start_indx_A[1:] - 1)],
        data_dict[a_b]['ipu_end_indices'][-1])
    turn_words_start_time_A = data_dict[a_b]['ipu_start_times'][turn_ipu_start_indx_A]
    turn_words_end_time_A = np.array(data_dict[a_b]['end_time_words'])[
        turn_words_end_indx_A]
    # turn_words_start_da_nite_A = np.array(data_dict[a_b]['da_nite_words'])[turn_words_start_indx_A]
    # turn_words_start_da_swbdType_A = np.array(data_dict[a_b]['da_swbdType_words'])[turn_words_start_indx_A]
    turn_one_da_bool_A = []
    # for strt, nd in zip(turn_words_start_indx_A, turn_words_end_indx_A):
    # turn_one_da_bool_A.append(bool(len(set(data_dict[a_b]['da_nite_words'][strt:nd])) - 1))
    turn_full_overlap = np.zeros(len(turn_ipu_start_indx_A))
    turn_full_overlap[np.array(turn_ipu_list_A)[full_overlap_list_A]] = 1
    turn_full_overlap = turn_full_overlap.astype(np.bool)

    data_dict[a_b]['turn_start_list'] = turn_ipu_list_A
    data_dict[a_b]['ipu_full_overlap'] = full_overlap_list_A
    data_dict[a_b]['turn_full_overlap'] = turn_full_overlap
    data_dict[a_b]['prev_gap_silence_bools'] = prev_gap_silence_bools_A
    data_dict[a_b]['turn_ipu_start_indx'] = turn_ipu_start_indx_A
    data_dict[a_b]['turn_words_start_indx'] = turn_words_start_indx_A
    data_dict[a_b]['turn_words_end_indx'] = turn_words_end_indx_A
    data_dict[a_b]['turn_words_start_time'] = turn_words_start_time_A
    data_dict[a_b]['turn_words_end_time'] = turn_words_end_time_A
    # data_dict[a_b]['turn_words_start_da_nite'] = turn_words_start_da_nite_A
    # data_dict[a_b]['turn_words_start_swbdType'] = turn_words_start_da_swbdType_A
    # data_dict[a_b]['turn_one_da_bool'] = turn_one_da_bool_A


# def get_das(a, data_dict, filename):
#   data_dict[a]['da_ISO'] = []
#   for ipu_strt, ipu_end in zip(data_dict[a]['ipu_start_indices'], data_dict[a]['ipu_end_indices']):
#     utt = ' '.join(data_dict[a]['target_words'][ipu_strt:ipu_end+1])
#     da = tagger.dialogue_act_tag(utt)
#     data_dict[a]['da_ISO'].append(da)

# def get_das(a, data_dict, filename):
#     utts = []
#     for ipu_strt, ipu_end in zip(data_dict[a]['ipu_start_indices'], data_dict[a]['ipu_end_indices']):
#         utt = ' '.join(data_dict[a]['target_words'][ipu_strt:ipu_end+1])
#         # da = tagger.dialogue_act_tag(utt)
#         utts.append([utt, []])
#     das = tagger.dialogue_act_tag_multi(deepcopy(utts))
#     data_dict[a]['da_ISO_first_pass_vec'] = das.astype(np.float16)
#     data_dict[a]['da_ISO_first_pass_class'] = tag_classes[np.argmax(
#         das, axis=1)]
#     das = tagger.dialogue_act_tag_multi(
#         utts, data_dict[a]['da_ISO_first_pass_class'])
#     data_dict[a]['da_ISO_second_pass_vec'] = das.astype(np.float16)
#     data_dict[a]['da_ISO_second_pass_class'] = tag_classes[np.argmax(
#         das, axis=1)]
#     das = tagger.dialogue_act_tag_multi(
#         utts, data_dict[a]['da_ISO_second_pass_class'])
#     data_dict[a]['da_ISO_third_pass_vec'] = das.astype(np.float16)
#     data_dict[a]['da_ISO_third_pass_class'] = tag_classes[np.argmax(
#         das, axis=1)]

#     return data_dict


# %% Create delayed frame annotations
t_1 = t.time()
# for i in range(0, len(files_feature_list)):
# for i in range(0, 10):
# tot_bad_nan = 0
data_dict_outer = {}
# for file in files_annotation_list:


def process_files(file):
    data_dict = {}
    for a_b in ['A', 'B']:
        for pa in os.walk(path_to_annotations):
            for in_file in pa[-1]:
                if (file in in_file) and ('word' in in_file) and (a_b in in_file):
                    word_file_path = os.path.join(pa[0], in_file)
                if (file in in_file) and ('trans' in in_file) and (a_b in in_file):
                    trans_file_path = os.path.join(pa[0], in_file)

        # # NOTE: uncomment this to get the transcription files for language modeling
        # line_list = []
        # with open(trans_file_path) as f:
        #   for line in f.readlines():
        #     word_line = line.strip().split()[3:]
        #     processed_line = ''
        #     for word in word_line:
        #       if not any([word in w for w in ['[silence]', '[noise]', '[laughter]', '[vocalized-noise]']]):
        #         while ('[' in word) or ('/' in word) or ('_' in word) or ('{' in word):
        #           word = deal_with_disfluency(word)
        #         processed_line += word + ' '
        #     if processed_line.strip():
        #       processed_line = processed_line.strip() + '\n'
        #       line_list.append(processed_line)
        #       # print(processed_line)
        #   processed_file = ''.join(line_list)
        # trans_file_out = open( path_to_output_trans_files + file + '.' + a_b + '.txt', 'w')
        # trans_file_out.write(processed_file)
        # trans_file_out.close()

        target_words, start_time, end_time = [], [], []
        with open(word_file_path) as f:
            for line in f.readlines():
                word_line = line.strip().split()
                # if not any([word_line[-1] in w for w in ['[silence]', '[noise]', '[laughter]', '[vocalized-noise]', '<b_aside>', '<e_aside>']]):
                if not any([word_line[-1] == w for w in ['[silence]', '[noise]', '[laughter]', '[vocalized-noise]', '<b_aside>', '<e_aside>']]):
                    start_time.append(float(word_line[1]))
                    raw_end = float(word_line[2])
                    end_time.append(
                        max([raw_end, start_time[-1]+frame_length]))
                    target_word = word_line[-1]
                    # if '_' in target_word:
                    #     print('')
                    while ('[' in target_word) or ('/' in target_word) or ('_' in target_word) or ('{' in target_word):
                        target_word = deal_with_disfluency(target_word)
                    target_words.append(target_word)

        # get ipus (ipu_count start at 1 rather than 0) !! wrong... changed to starting at 0!!
        ipu_bool = (np.array(start_time[1:]) -
                    np.array(end_time[:-1])) > ipu_thresh
        ipu_int = np.insert(ipu_bool.astype(np.int), 0, 1)
        ipu_inds = np.cumsum(ipu_int) - 1
        ipu_start_indices = np.where(ipu_int.astype(np.bool))[0]
        ipu_end_indices = np.concatenate(
            [ipu_start_indices[1:], [len(target_words)]]) - 1
        ipu_start_times = np.array(start_time)[ipu_int.astype(np.bool)]
        ipu_end_times = np.array(end_time)[ipu_end_indices]

        data_dict[a_b] = {}

        data_dict[a_b]['target_words'] = target_words
        data_dict[a_b]['start_time_words'] = start_time
        data_dict[a_b]['end_time_words'] = end_time
        # can be used to link words to ipus
        data_dict[a_b]['ipu_inds_words'] = ipu_inds

        # IPU annotations (dif length from word annotations. Indices index the words)
        data_dict[a_b]['ipu_start_indices'] = ipu_start_indices
        data_dict[a_b]['ipu_end_indices'] = ipu_end_indices
        data_dict[a_b]['ipu_start_times'] = ipu_start_times
        data_dict[a_b]['ipu_end_times'] = ipu_end_times

    # get turns
    ipu_starts_A = data_dict['A']['ipu_start_times']
    ipu_ends_A = data_dict['A']['ipu_end_times']
    ipu_starts_B = data_dict['B']['ipu_start_times']
    ipu_ends_B = data_dict['B']['ipu_end_times']
    prev_sil_start_A = np.insert(ipu_ends_A.copy()[:-1], 0, 0)
    prev_sil_end_A = ipu_starts_A.copy()
    prev_sil_start_B = np.insert(ipu_ends_B.copy()[:-1], 0, 0)
    prev_sil_end_B = ipu_starts_B.copy()

    get_turns_input_list_A = [ipu_starts_A, ipu_ends_A, prev_sil_start_A, prev_sil_end_A,
                              ipu_starts_B, ipu_ends_B, prev_sil_start_B, prev_sil_end_B]
    get_turns_input_list_B = [ipu_starts_B, ipu_ends_B, prev_sil_start_B, prev_sil_end_B,
                              ipu_starts_A, ipu_ends_A, prev_sil_start_A, prev_sil_end_A]
    get_ipus_turns('A', data_dict, get_turns_input_list_A)
    get_ipus_turns('B', data_dict, get_turns_input_list_B)

    get_updates_func('A', 'B', data_dict, file)
    get_updates_func('B', 'A', data_dict, file)

    # print(file + ' getting_das')
    # try:
    # get_das('A', data_dict, file)
    # get_das('B', data_dict, file)
    # except:
    # print('problem in file:' + file)
    print(file + ' done')
    # print('total_time: '+str(t.time()-t_1))
    return data_dict


# for file in files_annotation_list:
#   update_data = process_files(file)

# update_data = process_files(files_annotation_list[0])

if __name__ == '__main__':
    t_1 = t.time()
    pool = multiprocessing.Pool(processes=num_cores)
    result_list = pool.map(process_files, files_annotation_list)
    update_data = {key: value for key, value in zip(
        files_annotation_list, result_list)}
    # pickle.dump(update_data, open(
    #     './datasets/update_data_ms_state_full_200ms.p', 'wb'))
    pickle.dump(update_data, open(
        data_dir+'/update_data_ms_state_full_200ms.p', 'wb'))
    print(t.time() - t_1)

    print('done')
    # print('Max Length:' + str(max_len))
    print('total_time: '+str(t.time()-t_1))
