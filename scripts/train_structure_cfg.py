#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:19:17 2020

@editor: zhantao
"""

import os
import argparse

import json

class TrainOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        data = self.parser.add_argument_group('Data')
        data.add_argument('--graph_matrics_path', default='/home/logix/Documents/DEBOR/body_model/mesh_downsampling.npz', help='path to graph adjacency matrix and upsampling/downsampling matrices') 
        data.add_argument('--smpl_model_path', default='/home/logix/Documents/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', help='path to SMPL model') 
        data.add_argument('--smpl_objfile_path', default='/home/logix/Documents/DEBOR/body_model/text_uv_coor_smpl.obj', help='path to SMPL .obj file') 
        data.add_argument('--smpl_edges_path', default='/home/logix/Documents/DEBOR/body_model/edges_smpl.npy', help='path to SMPL edges(from CAPE).') 
        data.add_argument('--MGN_avgPose_path', default='/home/logix/Documents/DEBOR/body_model/MGN_train_avgPose.npy', help='path to the average pose of SMPL model in training set.')
        data.add_argument('--MGN_avgBeta_path', default='/home/logix/Documents/DEBOR/body_model/MGN_train_avgBetas.npy', help='path to the average shape of SMPL model in training set.')
        data.add_argument('--MGN_avgTrans_path', default='/home/logix/Documents/DEBOR/body_model/MGN_train_avgTrans.npy', help='path to the average translation of SMPL model in training set.')
        data.add_argument('--MGN_dispMeanStd_path', default='/home/logix/Documents/DEBOR/body_model/displacement_mean_std.npy', help='path to the mean and std of displacement.')
        data.add_argument('--img_per_object', default=4) 
        data.add_argument('--obj_usedImageIdx', nargs="+", default=[0], help='The image id to be used for training.') 

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--resume', dest='resume', default=True, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        gen.add_argument('--name', default='structure_ver1_full_singleEnc', help='Name of the experiment')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='pin memory to speedup training')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='do not pin memory')
        gen.set_defaults(pin_memory=True)
       
        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='../logs/local', help='Directory to store logs')
        io.add_argument('--mgn_dir', default='../datasets/MGN_brighter_augmented', help='Directory to the dataset')
        io.add_argument('--bgimg_dir', default='../datasets/ADE20K', help='Directory to backgound images')
        io.add_argument('--checkpoint', default=None, help='Path to save checkpoint')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained Graph CNN when starting training') 
        ss = io.add_mutually_exclusive_group()
        ss.add_argument('--save_aTestSample', dest='save_sample', action='store_true', help='save a sample during test')
        ss.add_argument('--no_save_aTestSample', dest='save_sample', action='store_false', help='do not save samples')
        io.set_defaults(save_sample=True)

        arch = self.parser.add_argument_group('Architecture')
        arch.add_argument('--model', default='United', choices=['United'], help='The model to be trained') 
        arch.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers') 
        arch.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN') 
        arch.add_argument('--latent_size', type=int, default=200, help='size of latent vector in sizernet') 
        arch.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding it in the network') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=100, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=4, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=256, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=5000, help='Checkpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency')
        train.add_argument('--num_downsampling', type=int, default=0, help='number of times downsampling the smpl mesh') 
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--noise_factor', type=float, default=0.4, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--scale_factor', type=float, default=0.25, help='rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]') 
        train.add_argument('--augmentation', dest='use_augmentation', default=True, action='store_false', help='Do augmentation') 
        train.add_argument('--augmentation_rgb', dest='use_augmentation_rgb', default=False, action='store_false', help='Do color jittering during training') 
        train.add_argument('--replace_background', dest='replace_background', default=False, action='store_false', help='Replace background of the rendered images.') 
        
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)
        
        # for the training of GCN, lr = 1e-3 would be enough, higher lr would not be very helpful
        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        optim.add_argument("--wd", type=float, default=1e-6, help="Weight decay weight")
        
        return 

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        
        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
        
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        
        self.save_dump()
        
        return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
    
