#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:41:39 2020

@author: zhantao
"""

structure_options = {
    # cut feature for diff tasks is more reasonable, as color
    # is invariant to pose, shape and disp; if learn it from 
    # the whole codes, might make the nn unstable.
    'cut_feature': False,    
    'inShape': 4,               # the H & W of latent 
    
    'structureList': {
        'SMPL': {
            'enable': True,
            'latent_shape': 512,   # all laten
            'latent_start': 0,
            'infeature': 154,       # 144 rots + 10 betas 
            'network': 'simple',    # or poseNet
            'shape': [256, 256, 512, 512]    # shape for MLP
            },
            
        'disp': {
            'enable': True,
            'latent_shape': 512,
            'latent_start': 512,
            'infeature': 20670,     # 6890x3, follow the paper
            'network': 'simple',    # or dispGraphNet
            'shape': [2048, 1024, 512, 512]    # shape for MLP
            },
        
        'text': {
            'enable': True,
            'latent_shape': 512,
            'latent_start': 1024,
            'infeature': 20670,     # 6890x3, follow the paper
            'network': 'simple-simple',    # -> vertices; downNet-upNet -> uv 
            'shape': [2048, 1024, 512, 512]    # shape for MLP
            },
        
        'camera':{
            'enable': True,
            'latent_shape': 512,
            'latent_start': 1536,
            'infeature': 3,     # 6890x3, follow the paper
            'network': 'simple',    # or cameraNet
            'shape': [16, 32, 64, 128]    # shape for MLP
            },
        },
    
    'graph_matrics_path' : '/home/zhantao/Documents/masterProject/DEBOR/body_model/mesh_downsampling.npz',
    'smpl_model_path' : '/home/zhantao/Documents/masterProject/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    
     }
