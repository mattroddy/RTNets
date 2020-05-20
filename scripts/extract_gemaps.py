# -*- coding: utf-8 -*-
import os
import sys
import time as t
from multiprocessing import Pool

num_workers = 7
data_dir = os.popen('sh ./paths.sh').read().rstrip()
audio_files_dir = data_dir + '/mono_dir/'
smile_path = './tools/opensmile-2.3.0/'
if len(sys.argv) == 2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0  # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    output_files_dir = data_dir+'/gemaps_features_50ms/'
    smile_command = smile_path + \
        'bin/linux_x64_standalone_static/SMILExtract -C ./tools/opensmile-2.3.0/config/gemaps_50ms/eGeMAPSv01a.conf'
else:
    output_files_dir = data_dir+'/gemaps_features_10ms/'
    smile_command = smile_path + \
        'bin/linux_x64_standalone_static/SMILExtract -C ./tools/opensmile-2.3.0/config/gemaps_10ms/eGeMAPSv01a.conf'

audio_files = os.listdir(audio_files_dir)

csv_file_list = [file.split('.')[0]+'.'+file.split('.')
                 [1]+'.csv' for file in audio_files]

if not(os.path.exists(output_files_dir)):
    os.mkdir(output_files_dir)

t_1 = t.time()
total_num_files = len(audio_files)


def loop_func_one(data):
    input_file, output_file = data
    print(input_file)
    os.system(smile_command + ' -I '+audio_files_dir +
              input_file+' -D '+output_files_dir+output_file)


if __name__ == '__main__':
    my_data = []
    for input_file, output_file in zip(audio_files, csv_file_list):
        my_data.append([input_file, output_file])
    p = Pool(num_workers)
    p.map(loop_func_one, my_data)
