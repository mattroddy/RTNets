import os
import platform

data_dir = os.popen('sh ./paths.sh').read().rstrip()
is_grove = 'Debian' in platform.uname()[3]
path_prefix = data_dir + '/'
path_prefix = '/home/matt/'
full_saved_data_fold = path_prefix + \
    'saved_datasets/full_valid150_noise_trnbatch_1_v3/'
print("dataset path: "+full_saved_data_fold)
dev_saved_data_fold = path_prefix+'saved_datasets/dev_set/'

fbanks_features_list = ['fbank_'+str(indx) for indx in range(40)]

gemaps_features_list_no_mfccs = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                                 'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                                 'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                                 'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                                 'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0']

gemaps_no_mfccs_50ms_dict_list = [
    {'folder_path': path_prefix+'gemaps_features_processed_50ms/znormalized',
     'features': gemaps_features_list_no_mfccs,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'gmaps50nomfccs',
     'dur_multiplier': 1,
     'stack_size': 1}]


fbanks_50ms_dict_list = [
    {
        'folder_path': path_prefix+'fbanks_processed_50ms/znormalized',
        'features': fbanks_features_list,
        'modality': 'acous',
        'is_h5_file': False,
        'uses_master_time_rate': True,
        'time_step_size': 1,
        'is_irregular': False,
        'short_name': 'fbanks50',
        'dur_multiplier': 1,
        'stack_size': 1
    }]
