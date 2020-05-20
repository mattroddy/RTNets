import pickle
import numpy as np

update_data = pickle.load(open('./datasets/update_data_200ms.p','rb'))

dataset_path = '../../saved_datasets/dev1/'
test = pickle.load(open(dataset_path+'test.p','rb'))


for t_idx, dat in enumerate(test['dataset']):
  f = dat['file'][0]
  a_usr = dat['a_usr'][0]
  if dat['associated_usr_turn'] != -1:
    sys_trn_strt_t = dat['update_strt_t'][0] + dat['y_strt_t'][0]
    sys_trn_end_t = dat['update_strt_t'][0] + dat['y_end_t'][0]
    usr_turn_end_time = update_data[f][a_usr]['turn_words_end_time'][dat['associated_usr_turn']]
    usr_turn_strt_time = update_data[f][a_usr]['turn_words_start_time'][dat['associated_usr_turn']]
    true_offset = sys_trn_strt_t - usr_turn_end_time
    if true_offset < -10 or true_offset > 10:
      print(t_idx)
      print(true_offset)
      print(f)
      print(a_usr)
      print('update_strt_t: {}'.format(dat['update_strt_t']))
      print('update_end_t: {}'.format(dat['update_end_t']))
      print('usr_trn_strt_t: {}'.format(usr_turn_strt_time))
      print('usr_trn_end_t: {}'.format(usr_turn_end_time))
      print('sys_strt_t: {}'.format(sys_trn_strt_t))
      print('sys_end_t: {}'.format(sys_trn_end_t))
      print()
