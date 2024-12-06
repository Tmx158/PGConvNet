import argparse
import torch
from exp.exp_imputation import Exp_Imputation
import random
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
parser = argparse.ArgumentParser(description='Model_PGConvNet')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Model_PGConvNet',
                    help='model name, options: [Model_PGConvNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')


# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


#Model_PGConvNet argument

parser.add_argument('--num_blocks', type=int, default=3, help='')
parser.add_argument('--large_size', type=int, default=3,help='')
parser.add_argument('--small_size', type=int, default=1,help='')
parser.add_argument('--kernel_size',type=int, default=3,help='')
parser.add_argument('--num_experts',type=int, default=3,help='')
parser.add_argument('--updim', type=int, default=128, help='dmodels in each stage')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')



# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--PatchTST_kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)


parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0,help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

#multi task
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
# short term
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
# classfication task
parser.add_argument('--class_dropout', type=float, default=0.05, help='classfication dropout')

#  BRITS
parser.add_argument('--rnn_hid_size', type=int, default=64, help='hidden size of RNN')


#SAITS
parser.add_argument('--saits_n_groups', type=int, default=2,
                    help="Number of groups in the SAITS model.")
parser.add_argument('--saits_n_group_inner_layers', type=int, default=1,
                    help="Number of inner layers in each group.")
parser.add_argument('--saits_d_time', type=int, default=96,
                    help="Size of the time dimension (set according to seq_len).")
parser.add_argument('--saits_d_feature', type=int, default=21,
                    help="Number of features (set according to feature_num).")
parser.add_argument('--saits_d_model', type=int, default=256,
                    help="Embedding dimension size for the model.")
parser.add_argument('--saits_d_inner', type=int, default=128,
                    help="Inner dimension size of the model.")
parser.add_argument('--saits_n_head', type=int, default=4,
                    help="Number of heads in the multi-head attention.")
parser.add_argument('--saits_d_k', type=int, default=64,
                    help="Dimension size of keys in attention.")
parser.add_argument('--saits_d_v', type=int, default=64,
                    help="Dimension size of values in attention.")
parser.add_argument('--saits_dropout', type=float, default=0.1,
                    help="Dropout rate for the model layers.")
parser.add_argument('--saits_input_with_mask', type=bool, default=True,
                    help="If True, concatenate input with mask along feature dimension.")
parser.add_argument('--saits_param_sharing_strategy', type=str, default="inner_group",
                    choices=["between_group", "inner_group"],
                    help="Parameter sharing strategy: 'between_group' shares parameters across groups, 'inner_group' shares within each group.")
parser.add_argument('--saits_MIT', type=bool, default=True,
                    help="If True, enables Masked Imputation Task (MIT) for additional imputation loss during training.")
parser.add_argument('--saits_device', type=str, default="cuda",
                    choices=["cpu", "cuda"],
                    help="Device to run the model on, 'cpu' or 'cuda'.")
parser.add_argument('--saits_diagonal_attention_mask', type=bool, default=True,
                    help="If True, applies a diagonal attention mask in self-attention layers.")





args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(f"{args.device_ids}")


print('Args in experiment:')
print(args)
if __name__ == '__main__':

    if args.task_name == 'imputation':
        Exp = Exp_Imputation

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_updim{}_nb{}_lk{}_sk{}_ker_size{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.updim,
                args.num_blocks,
                args.large_size,
                args.small_size,
                args.kernel_size,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training new  : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
