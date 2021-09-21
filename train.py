'''
Author: Ruizhi Liao

Main script to run training
'''

import os
import argparse
import logging

import torch

from resnet_chestxray.main_utils import ModelManager

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int,
                                        help='Mini-batch size')
parser.add_argument('--num_train_epochs', default=300, type=int,
                    help='Number of training epochs')
parser.add_argument('--loss_method', type=str,
                    default='CrossEntropyLoss',
                    help='Loss function for model training')
parser.add_argument('--init_lr', default=5e-4, type=float, 
                    help='Intial learning rate')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=1, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--data_dir', type=str,
                                        default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/',
                                        help='The image data directory')
parser.add_argument('--model_type', type=str,default='delta_change',
                   help='The directory for data. delta_change, percent_change, 2_class, 4_class')
# parser.add_argument('--dataset_metadata', type=str,
#                                         default=os.path.join(current_dir, 'data/percent_change/bnp_train.csv'),
#                                         help='The metadata for the model training ')
# parser.add_argument('--dataset_val_metadata', type=str,
#                                         default=os.path.join(current_dir, 'data/percent_change/bnp_validation.csv'),
#                                         help='The metadata for the model validation')
parser.add_argument('--save_dir', type=str,
        default='/data/vision/polina/projects/chestxray/ading/resnet_chestxray_experiments'
        )                               #default='/data/vision/polina/scratch/ruizhi/chestxray/experiments/'\
                                        #'supervised_image/tmp_postmiccai_v2/')
    
parser.add_argument('--label_key', type=str,
                    default='delta_bnp',
                    help='The supervised task (the key of the corresponding label column)')

parser.add_argument('--model_size', type=str, default='large', help='The size of the model')

parser.add_argument('--scheduler', dest='scheduler', action='store_true')

parser.set_defaults(scheduler=False)

# parser.add_argument('--train_type', type=str, default='regression',
#                    help='Type of model trained. regression or classification')

# parser.add_argument('--train_type', type=str, default='edema', help='Type of model to train. bnp, bnp+, creatinine, creatinine+, edema, all')


def train():
        args = parser.parse_args()
        args.dataset_metadata = os.path.join(current_dir, 'data/48h_window', args.model_type+'_adj', 'bnp_train.csv')
        args.dataset_val_metadata = os.path.join(current_dir, 'data/48h_window', args.model_type+'_adj', 'bnp_validation.csv')
        
        if args.model_size == 'small':
            args.model_architecture = args.model_architecture + '_small'
#         args.save_dir = '/data/vision/polina/projects/chestxray/ading/resnet_chestxray_experiments/' + args.model_type + '/48h_window/adj'
        if args.model_type == 'delta_change' or args.model_type =='percent_change':
            args.train_type = 'regression'
            args.output_channels = 1
        else:
            args.train_type = 'classification'
            if args.model_type == '2_class':
                args.output_channels = 2
            elif args.model_type == '3_class':
                args.output_channels = 3
            else:
                args.output_channels = 4
    
#         if args.train_type == 'bnp' or args.train_type =='creatinine':
#             args.output_channels = 1
#         elif args.train_type == 'all':
#             args.output_channels = 12 
#         else:
#             args.output_channels = 4
        print(args)
    
        '''
        Check cuda
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

        '''
        Create a sub-directory under save_dir 
        based on the label key
        '''
        args.save_dir = os.path.join(args.save_dir, 
                                                                 args.model_architecture+'_'+args.label_key)
        if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

        # Configure the log file
        log_path = os.path.join(args.save_dir, 'training.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w', 
                                                format='%(asctime)s - %(name)s %(message)s', 
                                                datefmt='%m-%d %H:%M')

        model_manager = ModelManager(model_name=args.model_architecture, 
                                                                 img_size=args.img_size,
                                                                 output_channels=args.output_channels,
                                     save_dir = args.save_dir)

        model_manager.train(device=device,
                                                args=args)

train()
