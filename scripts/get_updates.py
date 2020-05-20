import multiprocessing
import os
import pickle
import sys
import time as t
import xml.etree.ElementTree

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

# from DialogueAct_Tagger_master.corpora.Switchboard.DAMSL import DAMSL
from DAMSL import DAMSL
from swbd_to_damsl import swbd_to_damsl

# This file extracts the updates trom the nxt swbd annotations.
# To extract the annotations from the rest of the ms-state transcriptions use get_updates_msstate.py


# takes 1 min with 8 cores

"""
  *** New New Version !! ***
  I have got rid of the +-framelength 
  
  *** New version ***
  An IPU is part of speaker A's previous turn if there was no speech by speaker B during previous no-speech segment by that speaker A.
  Otherwise, the IPU starts a new turn.
  A turn by speaker A is also considered to be in full overlap if it ends/begins within +/- 1 * framelength of the beginning/end of an IPU by speaker B.
  Overlap definition: Any turn by A where there is a turn by B that starts earlier than (A_strt+framelength) and ends...
  after (A_end-framelength). We need the framelength term for the sake of rounding.
  Each system turn is associated with an offset from the user turn that began immediately prior to the beginning of the system turn.
  However, system turns that are in full overlap with a user turn are not associated with an offset.
  Each system turn is associated with a user turn. Multiple system turns may be associated with a single user turn.
  During testing we pad the user's silence with artificially generated silence to allow for late triggering of the system turn.
  K is the amount of frames prior to the ground truth point that the system encoder is presented with the system turn encoding.
  K is an integer value sampled from the uniform distribution on the interval [0, min(grd_trth_strt - update_strt_frame, grnd_trth_strt - usr_turn_strt , K_max)]
  An update is all of the frames by both speaker A and speaker B that span from the the end of the previous system turn to the end of the current system turn.

  Test points are all turn-start-points that belong to turns that are not in full overlap.
  A user_update is all the frames from the start of a test point to the start of the next test point.
  A system_update is all system turns that begin within the span from the start of user_update_start+frame_length to user_update_end+frame_length
  If the last system turn in a user_update ends before the start of the user_update the system_update ends
  The number of user_updates and system_updates for a given user/system assignment to a conversation must be the same.
  When the assignment is swapped i.e. the user becomes the system, the number of updates may change.
  The first user_update of a conversation is all the frames up until the first test point. (may include some user speech)

  *** Old version ***
  IPU is part of previous turn if there was no speech by the other speaker during previous no-speech segment by that person.
  Otherwise, the IPU starts a new turn.
  Test points are all turn-start-points that belong to turns that are not in full overlap.
  A user_update is all the frames from the start of a test point to the start of the next test point.
  A system_update is all system turns that begin within the span from the start of user_update_start+frame_length to user_update_end+frame_length
  If the last system turn in a user_update ends before the start of the user_update the system_update ends 
  The number of user_updates and system_updates for a given user/system assignment to a conversation must be the same.
  When the assignment is swapped i.e. the user becomes the system, the number of updates may change.
  The first user_update of a conversation is all the frames up until the first test point. (may include some user speech)
  """

num_cores = 8
ipu_thresh = 0.200  # 200ms threshold for ipu segmentation
frame_length = 0.050  # 50ms frame size
output_file_name = '/update_data_200ms.p'
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
path_to_features = data_dir+'/gemaps_features_processed_50ms/znormalized/'
path_to_annotations = data_dir + '/nxt_switchboard_ann/xml/phonwords/'
path_to_terminals = data_dir + '/nxt_switchboard_ann/xml/terminals/'
path_to_dialacts = data_dir + '/nxt_switchboard_ann/xml/dialAct/'
# path_to_extracted_annotations = './extracted_annotations/DA_advanced_50ms_raw/'

# if speed_setting == 0:
#     path_to_extracted_annotations = './extracted_annotations/DA_advanced_50ms_raw/'
# elif speed_setting == 1:
#     path_to_extracted_annotations = './extracted_annotations/DA_advanced_10ms_raw/'

# if not(os.path.exists(path_to_extracted_annotations)):
#     os.makedirs(path_to_extracted_annotations)


files_annotation_list = list(pd.read_csv(
    './splits/complete_nxt.txt', header=None, dtype=str)[0])

files_feature_list, files_output_list = [], []
files_terms, files_DA = [], []


def deal_with_disfluency(word):
    if word[0:9] == '[laughter':
        word = word[10:-1]

    if '/' in word:
        word = word[word.find('/') + 1:-1]

    if '_1' in word:
        word = word.replace('_1', '')

    if '<b_aside>' in word:
        word = word.replace('<b_aside>', '')

    if '<e_aside>' in word:
        word = word.replace('<e_aside>', '')

    if ('{' in word) or ('}' in word):
        if '{' in word:
            word = word.replace('{', '')
        if '}' in word:
            word = word.replace('}', '')

    if ('[' in word) or (']' in word):
        if '-[' in word:
            word = word.replace('-[', '')
            word = word.replace(']', '')
        elif ']-' in word:
            check_next_word_flag = 1
            word = word.replace(']-', '')
            word = word.replace('[', '')
        elif ']' in word:
            check_next_word_flag = 1
            word = word.replace(']', '')
            word = word.replace('[', '')
            word = word.replace('-', '')
        elif '[' in word:
            word = word.replace('[', '')
            word = word.replace(']', '')
            word = word.replace('-', '')
        else:
            print('unusual case: '+word)

    return word


def convert_to_ms_int(x):
    # converts float ms to rounded integers
    # use convention >= and <
    frame_length2 = frame_length * 100
    return np.int32(np.floor((np.array(x)*100 / frame_length2)) * frame_length2)


# %% First get vocabulary
no_change, added_to_end_list, max_len = 0, 0, 0
lengths_list, longer_than_one_list = [], []

# Get updates


def get_updates_func(a, b, data_dict, filename):
    max_sys_turns = 0

    sys_update_times = np.append([0.0], data_dict[b]['turn_words_end_time'])
    sys_update_end_times = sys_update_times[1:]
    sys_update_strt_times = sys_update_times[:-1]
    sys_update_turns = list(range(len(sys_update_strt_times)))
    updates = {}
    updates['sys_update_strt_times'] = sys_update_strt_times
    updates['sys_update_end_times'] = sys_update_end_times
    updates['sys_update_turns'] = sys_update_turns
    data_dict[a]['updates'] = updates


def get_turns(input_args):
    """
    Get turns for A
    """
    # convention: use >= and <
    input_args = [convert_to_ms_int(arg) for arg in input_args]

    ipu_starts_A, ipu_ends_A, prev_sil_start_A, prev_sil_end_A, \
        ipu_starts_B, ipu_ends_B, prev_sil_start_B, prev_sil_end_B = input_args

    # init lists
    turn_ipu_list = [0]
    # Init full_overlap_list
    full_overlap_list = [any((ipu_starts_B < ipu_starts_A[0])
                             & (ipu_ends_B >= ipu_ends_A[0]))]
    # boolean that indicates whether the gap preceding the current IPU is silent by both speakers (mutual silence). Needed for simulating the silence.
    other_speak_during_prev_sil = \
        any((ipu_ends_B >= prev_sil_start_A[0]) & (ipu_ends_B < prev_sil_end_A[0])) or \
        any((ipu_starts_B >= prev_sil_start_A[0]) & (ipu_starts_B < prev_sil_end_A[0])) or \
        any((ipu_starts_B < prev_sil_start_A[0])
            & (ipu_ends_B >= prev_sil_end_A[0]))
    prev_gap_silence_bools = [not(other_speak_during_prev_sil)]
    # prev_gap_silence_bools = [not(any((prev_sil_start_A[0] < ipu_ends_B) & (prev_sil_end_A[0] > ipu_ends_B)))]
    # can be full_overlap, partial_overlap(start), partial_overlap(end), no_overlap
    for strt_A, end_A, pre_sil_strt_A, pre_sil_end_A in zip(ipu_starts_A[1:], ipu_ends_A[1:], prev_sil_start_A[1:], prev_sil_end_A[1:]):
        # booleans to test if the utterance is in full overlap
        # if there is an IPU by B that ends after end+framelength of A's IPU and starts before start+framelength
        full_overlap = any((ipu_starts_B < strt_A)
                           & (ipu_ends_B >= end_A))
        full_overlap_list += [full_overlap]
        partial_overlaps_start = any((ipu_starts_B < strt_A)
                                     & (ipu_ends_B >= strt_A))
        other_speak_during_prev_sil = \
            any((ipu_ends_B >= pre_sil_strt_A) & (ipu_ends_B < pre_sil_end_A)) or \
            any((ipu_starts_B >= pre_sil_strt_A) & (ipu_starts_B < pre_sil_end_A)) or \
            any((ipu_starts_B < pre_sil_strt_A)
                & (ipu_ends_B >= pre_sil_end_A))

        prev_gap_silence_bools += [not(other_speak_during_prev_sil)]
        turn_ipu_list += [turn_ipu_list[-1] +
                          int(full_overlap or partial_overlaps_start or other_speak_during_prev_sil)]
    return turn_ipu_list, full_overlap_list, prev_gap_silence_bools


def get_ipus_turns(a_b, data_dict, get_turns_input_list):
    # turn_ipu_list_A, full_overlap_list_A, prev_gap_silence_bools_A = get_turns(*get_turns_input_list)
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
    turn_words_start_da_nite_A = np.array(data_dict[a_b]['da_nite_words'])[
        turn_words_start_indx_A]
    turn_words_start_da_damsl_A = np.array(data_dict[a_b]['da_damsl_words'])[
        turn_words_start_indx_A]
    turn_words_start_da_swbdType_A = np.array(data_dict[a_b]['da_swbdType_words'])[
        turn_words_start_indx_A]
    turn_words_end_da_nite_A = np.array(data_dict[a_b]['da_nite_words'])[
        turn_words_end_indx_A]
    turn_words_end_da_damsl_A = np.array(data_dict[a_b]['da_damsl_words'])[
        turn_words_end_indx_A]
    turn_words_end_da_swbdType_A = np.array(data_dict[a_b]['da_swbdType_words'])[
        turn_words_end_indx_A]

    turn_one_da_bool_A = []
    for strt, nd in zip(turn_words_start_indx_A, turn_words_end_indx_A):
        turn_one_da_bool_A.append(
            bool(len(set(data_dict[a_b]['da_damsl_words'][strt:nd])) == 1))
    turn_full_overlap = np.zeros(len(turn_ipu_start_indx_A))
    turn_full_overlap[np.array(turn_ipu_list_A)[full_overlap_list_A]] = 1
    turn_full_overlap = turn_full_overlap.astype(np.bool)

    ipu_one_da_bool_A = []
    for strt, nd in zip(data_dict[a_b]['ipu_start_indices'], data_dict[a_b]['ipu_end_indices']):
        ipu_one_da_bool_A.append(
            bool(len(set(data_dict[a_b]['da_damsl_words'][strt:nd])) == 1))
    # turn_full_overlap = np.zeros(len(turn_ipu_start_indx_A))
    # turn_full_overlap[np.array(turn_ipu_list_A)[full_overlap_list_A]] = 1
    # turn_full_overlap = turn_full_overlap.astype(np.bool)

    data_dict[a_b]['turn_start_list'] = turn_ipu_list_A
    data_dict[a_b]['ipu_full_overlap'] = full_overlap_list_A
    data_dict[a_b]['turn_full_overlap'] = turn_full_overlap
    data_dict[a_b]['prev_gap_silence_bools'] = prev_gap_silence_bools_A
    data_dict[a_b]['turn_ipu_start_indx'] = turn_ipu_start_indx_A
    data_dict[a_b]['turn_words_start_indx'] = turn_words_start_indx_A
    data_dict[a_b]['turn_words_end_indx'] = turn_words_end_indx_A
    data_dict[a_b]['turn_words_start_time'] = turn_words_start_time_A
    data_dict[a_b]['turn_words_end_time'] = turn_words_end_time_A
    data_dict[a_b]['turn_words_start_da_nite'] = turn_words_start_da_nite_A
    data_dict[a_b]['turn_words_start_da_damsl'] = turn_words_start_da_damsl_A
    data_dict[a_b]['turn_words_start_swbdType'] = turn_words_start_da_swbdType_A
    data_dict[a_b]['turn_words_end_da_nite'] = turn_words_end_da_nite_A
    data_dict[a_b]['turn_words_end_da_damsl'] = turn_words_end_da_damsl_A
    data_dict[a_b]['turn_words_end_swbdType'] = turn_words_end_da_swbdType_A
    data_dict[a_b]['turn_one_da_bool'] = turn_one_da_bool_A
    data_dict[a_b]['ipu_one_da_bool'] = ipu_one_da_bool_A


# %% Create delayed frame annotations
t_1 = t.time()
data_dict_outer = {}


def process_files(file):
    data_dict = {}
    for a_b in ['A', 'B']:
        e_words = xml.etree.ElementTree.parse(
            path_to_annotations + file + '.' + a_b + '.phonwords.xml').getroot()
        e_terms = xml.etree.ElementTree.parse(
            path_to_terminals + file + '.' + a_b + '.terminals.xml').getroot()
        e_DAs = xml.etree.ElementTree.parse(
            path_to_dialacts + file + '.' + a_b + '.dialAct.xml').getroot()
        parent_map_term = dict((c, p)
                               for p in e_terms.getiterator() for c in p)
        parent_map_DAs = dict((c, p) for p in e_DAs.getiterator() for c in p)
        target_words, start_time, end_time, da_nite, da_swbdType, da_id = \
            [], [], [], [], [], []
        da_damsl = []
        start_da_id, end_da_id = [], []
        prev_tag = 'uninterpretable'
        for atype in e_words.findall('phonword'):
            target_word = atype.get('orth')
            start_time.append(
                float(atype.get('{http://nite.sourceforge.net/}start')))
            # end_time.append(float(atype.get('{http://nite.sourceforge.net/}end')))
            raw_end = float(atype.get('{http://nite.sourceforge.net/}end'))
            end_time.append(max([raw_end, start_time[-1]+frame_length]))
            # target_msstate = atype.get('msstate')
            target_term_nite = atype.get('{http://nite.sourceforge.net/}id')
            while ('[' in target_word) or ('/' in target_word) or ('_' in target_word) or ('{' in target_word):
                target_word = deal_with_disfluency(target_word)
            target_words.append(target_word)
            try:
                terminal_node = parent_map_term[e_terms.find(".//word/{http://nite.sourceforge.net/}pointer[@href='" +
                                                             file + '.' + a_b + '.phonwords.xml' + "#id(" + target_term_nite + ")']")]
                target_dial_nite = terminal_node.attrib['{http://nite.sourceforge.net/}id']
                da_node = parent_map_DAs[e_DAs.find(".//da/{http://nite.sourceforge.net/}child[@href='" +
                                                    file + '.' + a_b + '.terminals.xml' + "#id(" + target_dial_nite + ")']")]
                da_attrib = da_node.attrib
                start_da_id.append(list(da_node)[0].attrib['href'])
                end_da_id.append(list(da_node)[-1].attrib['href'])
                da_nite.append(da_attrib['niteType'])
                da_swbdType.append(da_attrib['swbdType'])

                # if da_attrib['swbdType'] == '+':
                # da_damsl.append(da_damsl[-1])
                # else:
                # da_damsl.append(swbd_to_damsl[da_attrib['swbdType']])
                da_damsl.append(DAMSL.sw_to_damsl(
                    da_attrib['swbdType'], prev_tag))
                prev_tag = da_damsl[-1]

                da_id.append(da_attrib['{http://nite.sourceforge.net/}id'])

            except:
                da_nite.append('nan')
                da_swbdType.append('nan')
                da_id.append('nan')
                da_damsl.append('nan')
                start_da_id.append('nan')
                end_da_id.append('nan')
        # assert len(da_nite) == len(da_swbdType) == len(
            # da_id) == len(start_da_id) == len(start_time)

        ipu_bool = (np.array(start_time[1:]) -
                    np.array(end_time[:-1])) > ipu_thresh
        # ipu_bool = (np.array(convert_to_ms_int(start_time[1:])) - np.array(convert_to_ms_int(end_time[:-1]))) > convert_to_ms_int(ipu_thresh)
        ipu_int = np.insert(ipu_bool.astype(np.int), 0, 1)
        ipu_inds = np.cumsum(ipu_int) - 1
        ipu_start_indices = np.where(ipu_int.astype(np.bool))[0]
        ipu_end_indices = np.concatenate(
            [ipu_start_indices[1:], [len(target_words)]]) - 1
        ipu_start_times = np.array(start_time)[ipu_int.astype(np.bool)]
        ipu_end_times = np.array(end_time)[ipu_end_indices]

        # join DAs that are labeled as nan

        def fix_da_nans(da_vec):
            nans = np.where(np.array(da_vec) == 'nan')[0]
            loc_tot_bad_nan = 0
            for nan_ind in nans:
                target_ipu = ipu_inds[nan_ind]
                das = da_vec[ipu_start_indices[target_ipu]:ipu_end_indices[target_ipu] + 1]
                other_das = list(set(das) - set(['nan']))
                if len(other_das):
                    da_count = []
                    for da in other_das:
                        da_count.append(das.count(da))
                    sel_da = other_das[np.argmax(da_count)]
                    da_vec[nan_ind] = sel_da
                else:
                    loc_tot_bad_nan += 1
                return loc_tot_bad_nan

        bad_nan_count = fix_da_nans(da_nite)
        fix_da_nans(da_swbdType)
        fix_da_nans(da_id)
        fix_da_nans(start_da_id)
        fix_da_nans(end_da_id)
        fix_da_nans(da_damsl)

        data_dict[a_b] = {}

        # word annotations (all same len() )
        data_dict[a_b]['target_words'] = target_words
        data_dict[a_b]['start_time_words'] = start_time
        data_dict[a_b]['end_time_words'] = end_time
        data_dict[a_b]['end_da_id_words'] = end_da_id
        data_dict[a_b]['start_da_id_words'] = start_da_id
        data_dict[a_b]['da_nite_words'] = da_nite
        data_dict[a_b]['da_swbdType_words'] = da_swbdType
        data_dict[a_b]['da_damsl_words'] = da_damsl
        data_dict[a_b]['da_id_words'] = da_id
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
    print(file+' done')
    return data_dict


update_data = process_files(files_annotation_list[0])
print('')


# def get_plot_points(strts, ends, height):
#     xs, ys, = [], []
#     for strt, nd in zip(strts, ends):
#         xs += [strt, strt, nd, nd]
#         ys += [0, height, height, 0]
#     return xs, ys

# strt_t = 135.0
# end_t = 175.0

# ipu_strt_i_A = np.where((update_data['A']['ipu_start_times'] > strt_t) & (
#     update_data['A']['ipu_start_times'] < end_t))[0]
# ipu_strt_A_plt = update_data['A']['ipu_start_times'][ipu_strt_i_A]
# ipu_end_A_plt = update_data['A']['ipu_end_times'][ipu_strt_i_A]

# turn_strt_i_A = np.where((update_data['A']['turn_words_start_time'] > strt_t) & (
#     update_data['A']['turn_words_start_time'] < end_t))[0]
# turn_strt_A_plt = update_data['A']['turn_words_start_time'][turn_strt_i_A]
# turn_end_A_plt = update_data['A']['turn_words_end_time'][turn_strt_i_A]


# ipu_strt_i_B = np.where((update_data['B']['ipu_start_times'] > strt_t) & (
#     update_data['B']['ipu_start_times'] < end_t))[0]
# ipu_strt_B_plt = update_data['B']['ipu_start_times'][ipu_strt_i_B]
# ipu_end_B_plt = update_data['B']['ipu_end_times'][ipu_strt_i_B]

# turn_strt_i_B = np.where((update_data['B']['turn_words_start_time'] > strt_t) & (
#     update_data['B']['turn_words_start_time'] < end_t))[0]
# turn_strt_B_plt = update_data['B']['turn_words_start_time'][turn_strt_i_B]
# turn_end_B_plt = update_data['B']['turn_words_end_time'][turn_strt_i_B]

# ## plots

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,)


# turns_A_x, turns_A_y = get_plot_points(turn_strt_A_plt, turn_end_A_plt, height=1)
# ax1.fill_between(turns_A_x,0,turns_A_y)
# ipu_A_x, ipu_A_y = get_plot_points(ipu_strt_A_plt, ipu_end_A_plt, height=0.5)
# ax1.fill_between(ipu_A_x, 0, ipu_A_y)
# ax1.set_ylabel('turns ipus A')

# turns_B_x, turns_B_y = get_plot_points(turn_strt_B_plt, turn_end_B_plt, height=1)
# ax2.fill_between(turns_B_x, 0, turns_B_y)
# ipu_B_x, ipu_B_y = get_plot_points(ipu_strt_B_plt, ipu_end_B_plt, height=0.5)
# ax2.fill_between(ipu_B_x, 0, ipu_B_y)
# ax2.set_ylabel('turns ipus B')
# # plt.savefig('./myplt.eps')
# pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
# print('')


if __name__ == '__main__':
    t_1 = t.time()
    pool = multiprocessing.Pool(processes=num_cores)
    result_list = pool.map(process_files, files_annotation_list)
    # print(result_list)
    update_data = {key: value for key, value in zip(
        files_annotation_list, result_list)}
    pickle.dump(update_data, open(data_dir+output_file_name, 'wb'))
    # print('done')

    print(t.time() - t_1)
    # print('time for total:')
    # print(((t.time() - t_1) / 5) * len(files_annotation_list))
    print('done')
    print('Max Length:' + str(max_len))
    print('total_time: '+str(t.time()-t_1))
