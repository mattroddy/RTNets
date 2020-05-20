
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits


def test(in_dict):
    model = in_dict['model']
    test_dataloader = in_dict['test_dataloader']
    test_settings_dict = in_dict['test_settings_dict']
    dtype = in_dict['dtype']
    results_save = in_dict['results_save']
    results_save_output = in_dict['results_save_output']
    naming_dict = in_dict['naming_dict']
    data_count_dict = in_dict['data_count_dict']
    h5_dict = test_settings_dict['h5_dict']
    data_select_list = in_dict['data_select_list']
    g_f_str_to_idx = {g_f: idx for idx, g_f in enumerate(data_select_list)}
    pred_task_dict = in_dict['pred_task_dict']
    complete_file_list = in_dict['complete_file_list']
    model_in_length = test_settings_dict['model_in_length']
    output_order_test = test_settings_dict['output_order_test']
    time_bool_indices = test_settings_dict['time_bool_indices']
    if pred_task_dict['onset_test_flag']:
        train_results_dict = in_dict['train_results_dict']
    if test_settings_dict['final_test']:
        lvfci_dict = in_dict['lvfci_dict']

    losses_test_dict = {loss_task: []
                        for loss_task in pred_task_dict['loss_task_list']}
    batch_sizes, losses_l1 = [], []

    model.eval()
    model.change_seq_length(test_settings_dict['sequence_length_test'])
    if test_settings_dict['full_test']:
        # setup results_dict
        results_dict, losses_dict = {}, {}
        for pred_task in pred_task_dict['active_outputs']:
            loc_pred_length = pred_task_dict[pred_task]['pred_len']
            for file_name in test_settings_dict['test_file_list']:
                for g_f in data_select_list:
                    # create new arrays for the results
                    results_dict[os.path.join(
                        pred_task, file_name, g_f)] = np.zeros(
                            [test_settings_dict['results_lengths'][file_name], loc_pred_length]).squeeze()
                    # losses_dict[os.path.join(
                    #     pred_task, file_name, g_f)] = np.zeros(
                    #         [results_lengths[file_name], loc_pred_length])
    for batch_indx, batch in enumerate(test_dataloader):
        info_strt_indx = model_in_length + \
            len(pred_task_dict['active_outputs'])
        info_test = {str(k)[5:]: v for k, v in zip(
            output_order_test[info_strt_indx:], batch[info_strt_indx:])}
        if test_settings_dict['slow_test']:
            batch_length = int(torch.squeeze(
                info_test['batch_size']))  # check this
        else:
            batch_length = test_settings_dict['batch_size_fast_test']
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
            batch_length_prev = batch_length
        elif (batch_length_prev != batch_length):
            if test_settings_dict['slow_test']:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)
                model.init_hidden()
            batch_length_prev = batch_length

        model_input = []
        for model_indx in range(model_in_length):
            if model_indx in time_bool_indices:
                model_input.append(np.array(torch.squeeze(batch[model_indx]).transpose(1, 0)[
                                   :, :batch_length].numpy(), dtype=np.int))
            else:
                model_input.append(torch.squeeze(batch[model_indx]).transpose(
                    0, 1).transpose(0, 2).type(dtype))  # [:, :batch_length, :])

        y_dict = OrderedDict()
        for ti, task in enumerate(pred_task_dict['active_outputs']):
            # y_dict[task] = batch[4+ti].squeeze()[:batch_length].type(dtype)
            y_dict[task] = batch[model_in_length +
                                 ti].squeeze()[:batch_length].type(dtype)  # transpose?

        model_out = model(model_input)
        model_out_dict = {a: b.squeeze().transpose(0, 1) for a, b in zip(
            pred_task_dict['active_outputs'], model_out)}

        def true_approx_loss(out, y, count, num_data_points):
            y_mask = y != -1
            loss = binary_cross_entropy_with_logits(out[y_mask], y[y_mask])
            # scale HS loss_y_A_VA by number of instances
            loss_y_true = loss * (y_mask.sum().type(torch.float32) / count)
            loss_y_approx = loss * (dtype(
                np.array(num_data_points, dtype=np.float)) * y_mask.sum(
            ).type(dtype) / (dtype(
                np.array(count, dtype=np.float)) * y.shape[0] * y.shape[1]))
            return loss_y_true, loss_y_approx, loss

        loss_batch_dict = {}

        for pred_task in pred_task_dict['active_outputs']:
            if 'VA' in pred_task:
                loss_batch_dict[pred_task] = binary_cross_entropy_with_logits(
                    model_out_dict[pred_task], y_dict[pred_task])
            elif 'HS' in pred_task:
                true_approx = true_approx_loss(model_out_dict[pred_task], y_dict[pred_task],
                                               data_count_dict['cont_hold_shift_count'],
                                               data_count_dict['num_data_points'])
                loss_batch_dict[pred_task + '_true'] = true_approx[0]
                loss_batch_dict[pred_task + '_approx'] = true_approx[1]
                y_tmp = y_dict[pred_task] != -1

            elif 'BC' in pred_task:
                true_approx = true_approx_loss(model_out_dict[pred_task], y_dict[pred_task],
                                               data_count_dict['bc_count'],
                                               data_count_dict['num_data_points'])
                loss_batch_dict[pred_task + '_true'] = true_approx[0]
                loss_batch_dict[pred_task + '_approx'] = true_approx[1]

        # loss_batch_dict['combined'] = torch.tensor(0.0).cuda()
        loss_batch_dict['combined'] = torch.tensor(0.0)
        loss_task_list_temp = [x for x in pred_task_dict['loss_task_list'] if (
            x != 'combined') and not('true' in x)]
        for loss_task in loss_task_list_temp:  # get without combined
            if torch.isnan(loss_batch_dict[loss_task]):
                loss_batch_dict['combined'] += 0.0
            else:
                loss_batch_dict['combined'] += pred_task_dict[loss_task[:4]]['weight'] \
                    * loss_batch_dict[loss_task]
        # loss_no_reduce = loss_func_L1_no_reduce(model_out_dict['A_VA'], y_dict['A_VA'])
        # loss_l1 = loss_func_L1(model_out_dict['A_VA'], y_dict['A_VA'])

        for loss_task in losses_test_dict.keys():
            if torch.isnan(loss_batch_dict[loss_task]):
                losses_test_dict[loss_task].append(0.0)
            else:
                losses_test_dict[loss_task].append(
                    loss_batch_dict[loss_task].data.cpu().numpy())
        # losses_l1.append(loss_l1.data.cpu().numpy())
        batch_sizes.append(batch_length)

        if test_settings_dict['set_type'] == 'test':
            file_name_list = [
                complete_file_list[int(np.squeeze(
                    np.array(info_test['file_names']))[i])]
                for i in range(batch_length)
            ]
            gf_name_list = [
                data_select_list[int(np.squeeze(
                    np.array(info_test['g_f']))[i])]
                for i in range(batch_length)
            ]
            time_index_list = [
                np.array(torch.squeeze(info_test['time_indices'])[i])
                for i in range(batch_length)
            ]
        else:
            file_name_list = info_test['file_names']
            gf_name_list = info_test['g_f']
            time_index_list = info_test['time_indices']

        if test_settings_dict['full_test']:
            for pred_task in pred_task_dict['active_outputs']:
                sig_out = F.sigmoid(
                    model_out_dict[pred_task]).data.cpu().numpy()
                for file_name, g_f_indx, time_indices, loc_batch_indx in zip(
                        file_name_list, gf_name_list, time_index_list,
                        range(batch_length)):
                    results_dict[os.path.join(pred_task, file_name, g_f_indx)][int(time_indices[0]): int(time_indices[1])] = \
                        sig_out[loc_batch_indx]
                    # F.sigmoid(model_out_dict[pred_task][loc_batch_indx]).data.cpu().numpy()

                    # F.sigmoid(model_out_dict[pred_task]).data.cpu().numpy() # this can probably be sped up

    for loss_task, loss in losses_test_dict.items():
        if any(x in loss_task for x in ['VA', 'combined']):
            results_save[test_settings_dict['result_type'] + '_losses_' + loss_task].append(
                np.sum(np.array(batch_sizes) * np.array(loss)) / np.sum(np.array(batch_sizes)))
        elif 'approx' in loss_task:
            results_save[test_settings_dict['result_type'] +
                         '_losses_' + loss_task].append(np.mean(loss))
        elif 'true' in loss_task:
            results_save[test_settings_dict['result_type'] +
                         '_losses_' + loss_task].append(np.sum(loss))
    # loss_weighted_mean_l1 = np.sum(np.array(batch_sizes) * np.array(losses_l1)) \
    #     / np.sum(batch_sizes)

    if not test_settings_dict['full_test']:
        # return loss_weighted_mean_combined
        return results_save[test_settings_dict['result_type']+'_losses_combined'][-1]
    else:
        if pred_task_dict['A_VA']['pred_len'] >= 20:
            # get hold-shift f-scores
            get_hold_shift(pred_task_dict, test_settings_dict, h5_dict, results_save,
                           data_select_list, results_dict)
            # get prediction at overlap f-scores
            get_pred_overlap(pred_task_dict, test_settings_dict, h5_dict,
                             results_save, data_select_list, results_dict)

        # get continuous hold-shift f-scores
        get_cont_hold_shift(pred_task_dict, test_settings_dict, h5_dict,
                            results_save, data_select_list, results_dict)

        # get prediction at onset f-scores
        if pred_task_dict['onset_test_flag']:
            get_pred_onset(pred_task_dict, test_settings_dict, h5_dict,
                           results_save, data_select_list, results_dict,
                           train_results_dict)

        if test_settings_dict['final_test'] and pred_task_dict['A_VA']['bool'] \
                and lvfci_dict['lvfci_bool']:
            # print('get pomdp outputs')
            # get_pomdp(pred_task_dict, test_settings_dict, h5_dict,
            #           results_save, data_select_list, results_dict, lvfci_dict)
            print('get latency vs false cut in')
            get_lvfci(pred_task_dict, test_settings_dict, h5_dict,
                      results_save, data_select_list, results_dict, lvfci_dict)

        return results_save[test_settings_dict['result_type'] + '_losses_combined'][-1], results_dict
