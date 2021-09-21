'''
Author: Ruizhi Liao

Main script to run evaluation
'''

import os
import argparse
import logging
import json

import torch

from resnet_chestxray.main_utils import ModelManager, build_model

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int,
                                        help='Mini-batch size')

parser.add_argument('--label_key', default='delta_bnp', type=str,
                                        help='The label key/classification task')

parser.add_argument('--img_size', default=256, type=int,
                    help='The size of the input image')
parser.add_argument('--output_channels', default=4, type=int,
                    help='The number of ouput channels')
parser.add_argument('--model_architecture', default='resnet256_6_2_1', type=str,
                    help='Neural network architecture to be used')

parser.add_argument('--data_dir', type=str,
                                        default='/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit_256/',
                                        help='The image data directory')
parser.add_argument('--model_type', type=str,default='delta_change',
                   help='The directory for data. delta_change, percent_change, 2_class, 4_class')

# parser.add_argument('--dataset_metadata', type=str,
#                                         default=os.path.join(current_dir, 'data/test_edema.csv'),
#                                         help='The metadata for the model training ')
parser.add_argument('--save_dir', type=str, default = '/data/vision/polina/projects/chestxray/ading/resnet_chestxray_experiments')
#                                       default='/data/vision/polina/scratch/ruizhi/chestxray/experiments/supervised_image/'\
#                                        'tmp_postmiccai_v2/')
parser.add_argument('--checkpoint_name', type=str,
                                        default='pytorch_model_epoch300.bin')

parser.add_argument('--model_size', type = str, default='large', help='The size of the model')

# parser.add_argument('--train_data', dest = 'train', action='store_true')

# parser.add_argument('--eval_data', dest = 'train', action='store_false')

# parser.set_defaults(train=False)


def eval(all_epochs=-1):
        args = parser.parse_args()
        
        args.dataset_metadata = os.path.join(current_dir, 'data/48h_window', args.model_type, 'bnp_test.csv')
        
        args.save_dir = '/data/vision/polina/projects/chestxray/ading/resnet_chestxray_experiments/' + args.model_type + '/48h_window'
#         args.dataset_metadata = os.path.join(current_dir, 'data/48h_window', args.model_type+'_adj', 'bnp_test.csv')
        
#         args.save_dir = '/data/vision/polina/projects/chestxray/ading/resnet_chestxray_experiments/' + args.model_type + '/48h_window/adj'
        
        
        if args.model_size == 'small':
            args.model_architecture = args.model_architecture + "_small"
            
        if args.model_type == 'delta_change' or args.model_type =='percent_change':
            args.train_type = 'regression'
            args.output_channels = 1
        else:
            args.train_type = 'classification'
            if args.model_type == '2_class':
                args.output_channels = 2
            elif args.model_type=='3_class':
                args.output_channels = 3
            else:
                args.output_channels = 4
        
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
        args.save_dir = os.path.join(args.save_dir, args.model_architecture+'_'+args.label_key)
        
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)

        model_manager = ModelManager(model_name=args.model_architecture, 
                                     img_size=args.img_size,
                                     save_dir = args.save_dir,
                                     output_channels=args.output_channels)
        inference_results, eval_results= model_manager.eval(device=device,
                                                                                                                args=args,
                                                                                                                checkpoint_path=checkpoint_path)

        print(f"{checkpoint_path} evaluation results: {eval_results}")

        '''
        Evaluate on all epochs if all_epochs>0
        '''
        if all_epochs>0:
                aucs_all_epochs = []
                for epoch in range(all_epochs):
                        args.checkpoint_name = f'pytorch_model_epoch{epoch+1}.bin'
                        checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)
                        model_manager = ModelManager(model_name=args.model_architecture,
                                                                                 img_size=args.img_size,
                                                                                 output_channels=args.output_channels)
                        inference_results, eval_results= model_manager.eval(device=device,
                                                                                                                                args=args,
                                                                                                                                checkpoint_path=checkpoint_path)
                        if args.label_key == 'edema_severity':
                                aucs_all_epochs.append(eval_results['ordinal_aucs'])
                        else:
                                aucs_all_epochs.append(eval_results['aucs'][0])

                print(f"All epochs AUCs: {aucs_all_epochs}")

                eval_results_all={}
                eval_results_all['ordinal_aucs']=aucs_all_epochs
                results_path = os.path.join(args.save_dir, 'eval_results_all.json')
                with open(results_path, 'w') as fp:
                        json.dump(eval_results_all, fp)

eval(all_epochs=-1)
