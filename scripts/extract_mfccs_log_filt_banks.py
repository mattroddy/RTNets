# -*- coding: utf-8 -*-
import os
import sys
import time as t
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc

# Takes about 30mins for switchboard nxt with 12 workers
num_workers = 8
t_0 = t.time()

data_dir = os.popen('sh ./paths.sh').read().rstrip()
path_to_wav_dir = data_dir + '/mono_dir/'

# path_to_wav_dir = '../../dialogues_mono'
if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0  # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    # path_to_output_mfccs_dir = '../../mfccs_50ms'
    path_to_output_fbanks_dir = data_dir+'/fbanks_50ms'
    # path_to_processed_mfccs_dir = '../../mfccs_processed_50ms'
    path_to_processed_fbanks_dir = data_dir+'/fbanks_processed_50ms'
    frame_length = 0.05
    frame_stride = 0.05
    shift_back_amount = 0
    fft_length = 1024
else:
    # path_to_output_mfccs_dir = './signals/mfccs_10ms'
    path_to_output_fbanks_dir = data_dir+'/fbanks_10ms'
    # path_to_processed_mfccs_dir = './signals/mfccs_processed_10ms'
    path_to_processed_fbanks_dir = data_dir+'/fbanks_processed_10ms'
    frame_length = 0.02
    frame_stride = 0.01
    shift_back_amount = 1
    fft_length = 512

num_filts = 40
ceplifter = 27
pre_emphasis = 0.97


file_list = os.listdir(path_to_wav_dir)

file_list = [fn for fn in file_list if fn.endswith('.wav')]
csv_file_list = [file.split('.')[0]+'.'+file.split('.')
                 [1]+'.csv'for file in file_list]

# %% Loop through files and extract features
# if not(os.path.exists(path_to_output_mfccs_dir)):
# os.mkdir(path_to_output_mfccs_dir)

if not(os.path.exists(path_to_output_fbanks_dir)):
    os.mkdir(path_to_output_fbanks_dir)

mfcc_names, fbank_names = [], []
for n_i in range(13):
    mfcc_names.append('mfcc_'+str(n_i))
for n_i in range(num_filts):
    fbank_names.append('fbank_'+str(n_i))

mean_mfcc_array, std_mfcc_array, mean_fbank_array, std_fbank_array, max_fbank_array, min_fbank_array, num_vals = \
    [], [], [], [], [], [], []


def extract_feats(in_dat):
    file, csv_file = in_dat
    print('processing file: '+file)
    (rate, sig) = wav.read(os.path.join(path_to_wav_dir, file))

    # mfcc_feat = mfcc(sig, rate, frame_length, frame_stride, nfilt=num_filts,
    # nfft=fft_length, preemph=pre_emphasis, ceplifter=ceplifter)
    fbank_feat = logfbank(sig, rate, frame_length, frame_stride,
                          nfilt=num_filts, nfft=fft_length,
                          preemph=pre_emphasis)

    # mean_mfcc = np.mean(mfcc_feat, axis=0)
    # std_mfcc = np.std(mfcc_feat, axis=0)
    mean_fbank = np.mean(fbank_feat, axis=0)
    std_fbank = np.std(fbank_feat, axis=0)
    max_fbank = np.max(fbank_feat, axis=0)
    min_fbank = np.min(fbank_feat, axis=0)

    # num_vals.append( len(mfcc_feat))
    # num_val = len(mfcc_feat)

    num_vals.append(len(fbank_feat))
    num_val = len(fbank_feat)

    # pd.DataFrame(mfcc_feat, columns=mfcc_names).to_csv(os.path.join(path_to_output_mfccs_dir, csv_file),
    # sep=',', index=False)
    pd.DataFrame(fbank_feat, columns=fbank_names).to_csv(os.path.join(path_to_output_fbanks_dir, csv_file),
                                                         sep=',', index=False)
    # return mean_mfcc, std_mfcc, mean_fbank, std_fbank, max_fbank, min_fbank, num_val
    return mean_fbank, std_fbank, max_fbank, min_fbank, num_val
# for file,csv_file in zip(file_list,csv_file_list):


def write_files(in_dat):

    file, csv_file = in_dat
    print('file: '+file)
    # annotation=pd.read_csv( os.path.join(path_to_annotations,file.split('.')[0]+'.'+file.split('.')[1]+'.csv'),delimiter = ',')

    # Shift back features by a given window size
    # raw feats
    # mfcc_feats_raw = np.array(pd.read_csv(
    # os.path.join(path_to_output_mfccs_dir, csv_file)))
    fbank_feats_raw = np.array(pd.read_csv(
        os.path.join(path_to_output_fbanks_dir, csv_file)))
    # frame_times = pd.DataFrame(np.concatenate([ np.zeros(shift_back_amount),annotation.frameTime]),columns=['frame_time'])
    # frame_times = pd.DataFrame(np.arange(
    #     0, mfcc_feats_raw.shape[0]+shift_back_amount) * frame_stride, columns=['frame_time'])
    frame_times = pd.DataFrame(np.arange(
        0, fbank_feats_raw.shape[0]+shift_back_amount) * frame_stride, columns=['frame_time'])
    # mfcc_pd = pd.DataFrame(np.concatenate([np.zeros(
    # mfcc_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, mfcc_feats_raw.shape[1]]), mfcc_feats_raw]), columns=mfcc_names)
    # fbank_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, fbank_feats_raw.shape[1]]), fbank_feats_raw]), columns=fbank_names)
    # pd.concat([frame_times, mfcc_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_mfccs_dir, 'raw', csv_file), sep=',', index=False)
    # pd.concat([frame_times, fbank_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_fbanks_dir, 'raw', csv_file), sep=',', index=False)

    # znormalized
    # mfcc_feats = (mfcc_feats_raw - np.mean(mfcc_feats_raw, 0)) / \
    #     np.std(mfcc_feats_raw, 0)
    fbank_feats = (fbank_feats_raw - np.mean(fbank_feats_raw, 0)
                   ) / np.std(fbank_feats_raw, 0)
    # mfcc_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, mfcc_feats.shape[1]]), mfcc_feats]), columns=mfcc_names)
    fbank_pd = pd.DataFrame(np.concatenate([np.zeros(
        [shift_back_amount, fbank_feats.shape[1]]), fbank_feats]), columns=fbank_names)
    # pd.concat([frame_times, mfcc_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_mfccs_dir, 'znormalized', csv_file), sep=',', index=False)
    pd.concat([frame_times, fbank_pd], axis=1).to_csv(os.path.join(
        path_to_processed_fbanks_dir, 'znormalized', csv_file), sep=',', index=False)

    # # znormalized_pooled
    # mfcc_feats = (mfcc_feats_raw - pooled_mean_mfcc) / pooled_variance_mfcc
    # fbank_feats = (fbank_feats_raw - pooled_mean_fbank) / pooled_variance_fbank
    # mfcc_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, mfcc_feats.shape[1]]), mfcc_feats]), columns=mfcc_names)
    # fbank_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, fbank_feats.shape[1]]), fbank_feats]), columns=fbank_names)
    # pd.concat([frame_times, mfcc_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_mfccs_dir, 'znormalized_pooled', csv_file), sep=',', index=False)
    # pd.concat([frame_times, fbank_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_fbanks_dir, 'znormalized_pooled', csv_file), sep=',', index=False)

    # # min_max fbanks
    # fbank_feats = (fbank_feats_raw - min_fbank_array) / \
    #     (max_fbank_array - min_fbank_array)
    # fbank_pd = pd.DataFrame(np.concatenate([np.zeros(
    #     [shift_back_amount, fbank_feats.shape[1]]), fbank_feats]), columns=fbank_names)
    # pd.concat([frame_times, fbank_pd], axis=1).to_csv(os.path.join(
    #     path_to_processed_fbanks_dir, 'min_max', csv_file), sep=',', index=False)


# %%  Main
if __name__ == '__main__':
    extract_feats_data = []
    for target_file, annotation_file in zip(file_list, csv_file_list):
        extract_feats_data.append([target_file, annotation_file])
    p = Pool(num_workers)
    multi_output = p.map(extract_feats, extract_feats_data)

    mean_fbank_array, std_fbank_array, max_fbank_list, min_fbank_list, num_vals = \
        [], [], [], [], []

    # for l in multi_output:
    #     mean_mfcc_array.append(l[0])
    #     std_mfcc_array.append(l[1])
    #     mean_fbank_array.append(l[2])
    #     std_fbank_array.append(l[3])
    #     max_fbank_list.append(l[4])
    #     min_fbank_list.append(l[5])
    #     num_vals.append(l[6])

    for l in multi_output:
        mean_fbank_array.append(l[0])
        std_fbank_array.append(l[1])
        max_fbank_list.append(l[2])
        min_fbank_list.append(l[3])
        num_vals.append(l[4])

    min_fbank_array = np.min(np.array(min_fbank_list), axis=0)
    max_fbank_array = np.max(np.array(max_fbank_list), axis=0)

    min_fbank_array = np.min(np.array(min_fbank_list), axis=0)
    max_fbank_array = np.max(np.array(max_fbank_list), axis=0)

    # %% calculate pooled variance and mean for mfccs

    # numerator_variance_mfcc = np.sum(np.tile(np.array(
    #     num_vals) - 1, [13, 1]).transpose() * np.array(std_mfcc_array) ** 2, axis=0)
    # pooled_variance_mfcc = np.sqrt(
    #     numerator_variance_mfcc / (sum(num_vals) - len(num_vals)))
    # numerator_mean_mfcc = np.sum(np.tile(
    #     np.array(num_vals), [13, 1]).transpose() * np.array(mean_mfcc_array), axis=0)
    # pooled_mean_mfcc = numerator_mean_mfcc / (sum(num_vals))

    numerator_variance_fbank = np.sum(
        np.tile(np.array(num_vals) - 1, [num_filts, 1]).transpose() * np.array(std_fbank_array) ** 2, axis=0)
    pooled_variance_fbank = np.sqrt(
        numerator_variance_fbank / (sum(num_vals) - len(num_vals)))
    numerator_mean_fbank = np.sum(
        np.tile(np.array(num_vals), [num_filts, 1]).transpose() * np.array(mean_fbank_array), axis=0)
    pooled_mean_fbank = numerator_mean_fbank / (sum(num_vals))

    # %% reprocess the features: normalize using pooled variance and means, use min/max scaling and pooled variance for filter banks

    # if not (os.path.exists(path_to_processed_mfccs_dir)):
    #     os.mkdir(path_to_processed_mfccs_dir)
    #     os.mkdir(os.path.join(path_to_processed_mfccs_dir, 'znormalized'))
    #     os.mkdir(os.path.join(path_to_processed_mfccs_dir, 'znormalized_pooled'))
    #     os.mkdir(os.path.join(path_to_processed_mfccs_dir, 'raw'))

    if not (os.path.exists(path_to_processed_fbanks_dir)):
        os.mkdir(path_to_processed_fbanks_dir)
        os.mkdir(os.path.join(path_to_processed_fbanks_dir, 'znormalized'))
        # os.mkdir(os.path.join(path_to_processed_fbanks_dir, 'znormalized_pooled'))
        # os.mkdir(os.path.join(path_to_processed_fbanks_dir, 'raw'))
        # os.mkdir(os.path.join(path_to_processed_fbanks_dir, 'min_max'))

    # %% Reprocess files
    print('reprocessing files')
    p = Pool(num_workers)
    multi_output = p.map(write_files, extract_feats_data)


print('total time taken: {}'.format(t.time()-t_0))
