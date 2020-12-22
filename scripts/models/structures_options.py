#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:41:39 2020

@author: zhantao
"""

structure_options = {
    # cut feature for diff tasks is more reasonable, as color
    # is invariant to pose, shape and disp; if learn it from 
    # the whole codes, might make the nn unstable. Besides,
    # 1024 channels are reserved for the image feature.
    'cut_feature': False,    
    'inShape': 4,               # the H & W of latent 
    
    'structureList': {
        
        'indexMap': {
            'network': 'simple',    # 
            'supVis_lossFunc' : 'L1',   # ('L1', 'L2') 
            'batch_lossFunc'  : 'L2',   # only support L2 yet
            
            'weight' : 0,
            'normalization': True,    # using normalized/standard L1 loss 
            'normlizeThres': 0.005    # threshold should < min gap 0.01887/2 
            },
        
        'SMPL': {
            'enable': True,
            'latent_shape': 512,
            'latent_start': 0,
            'infeature': 164,       # 144 rots + 10 betas + 3 trans + 7 camera
            'network': 'iterNet',    # or iterNet and will overwrite camera
            'shape': [256, 256, 512, 512],    # shape for MLP
            'actFunc': False,       # if enable activation function in network
             
            'latent_lossFunc' : 'L2',   # ('L1', 'L2', 'cross_entropy') 
            'supVis_lossFunc' : 'L1',   # ('L1', 'L2') 
            'batch_lossFunc'  : 'L2',   # only support L2 yet
            
            'weight' : {
                'latentCode'  : 0,
                'supervision' : 1
                }, 
            },
            
        'disp': {
            'enable': False,
            'latent_shape': 384,
            'latent_start': 512,
            'infeature': 20670,     # 6890x3, follow the paper
            'network': 'simple',    # or dispGraphNet
            'shape': [2048, 1024, 512, 384],    # shape for MLP
            
            'latent_lossFunc' : 'L2',   # ('L1', 'L2', 'cross_entropy') 
            'supVis_lossFunc' : 'L1',   # ('L1', 'L2') 
            'batch_lossFunc'  : 'L2',   # only support L2 yet
            
            'weight' : {
                'latentCode'  : 0.001,
                'supervision' : 0.1
                }, 
            },
        
        'text': {
            'enable': False,
            'latent_shape': 512,
            'latent_start': 1024,
            'infeature': 20670,     # 6890x3, follow the paper
            'network': 'simple-simple',    # -> vertices; downNet-upNet -> uv 
            'shape': [2048, 1024, 512],    # shape for MLP
            
            'latent_lossFunc' : 'L2',   # ('L1', 'L2', 'cross_entropy') 
            'supVis_lossFunc' : 'L1',   # ('L1', 'L2') 
            'batch_lossFunc'  : 'L2',    # only support L2 yet
            
            'weight' : {
                'latentCode'  : 0.001,
                'supervision' : 0.1
                }, 
            },
        
        'camera':{
            'enable': False,
            'latent_shape': 128,
            'latent_start': 896,
            'infeature': 7,     
            'shape': [16, 32, 64],    # shape for MLP
            
            'latent_lossFunc' : 'L2',   # ('L1', 'L2', 'cross_entropy') 
            'supVis_lossFunc' : 'L1',   # ('L1', 'L2') 
            'batch_lossFunc'  : 'L2',    # only support L2 yet
            
            'weight' : {
                'latentCode'  : 0.001,
                'supervision' : 0.1
                }, 
            },
        
        'rendering':{
            'enable': False,
            'downsample' : True,        # fit a subset to avoid local minima
            'sampleRate' : 0.1,         # the subset rate
            'proj_lossFunc'  : 'L1',    # ('L1', 'L2') 
            'render_lossFunc': 'L1',    # ('L1', 'L2') 

            'weight' : {
                'proj_weight': 0.001,
                'render_weight': 0.001
                }
            }
        },
    
    'graph_matrics_path' : 
        '/home/zhantao/Documents/masterProject/DEBOR/body_model/mesh_downsampling.npz',
    'smpl_model_path' : 
        '/home/zhantao/Documents/masterProject/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    'smpl_objfile_path' :
        '/home/zhantao/Documents/masterProject/DEBOR/body_model/text_uv_coor_smpl.obj'
     }
