#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:05:02 2020

@author: zhantao
"""
from os.path import abspath
import sys
if __name__ == '__main__':
    if abspath('../') not in sys.path:
        sys.path.append(abspath('../'))
from collections import namedtuple

import torch
import torch.nn as nn

from structures_options import structure_options
from subtasks_nn import simpleMLP, cameraNet
from subtasks_nn import poseNet, upNet, downNet, dispGraphNet


class DEBORNet(nn.Module):
    
    def __init__(self, structure_options, A = None, ref_vertices = None, 
                 device = 'cuda'):
        super(DEBORNet, self).__init__()
        
        self.options = structure_options
        self.infeature = 2048
        
        # --------- SMPL model encoder and decoder ---------
        if options.structureList.SMPL.enable:
            self.SMPLenc = simpleMLP(
                self.options.structureList.SMPL.infeature, 
                self.options.structureList.SMPL.latent_shape,
                self.options.structureList.SMPL.shape)
            if options.structureList.SMPL.network == 'simple':
                self.SMPLdec = simpleMLP(
                    self.options.structureList.SMPL.latent_shape,
                    self.options.structureList.SMPL.infeature,
                    self.options.structureList.SMPL.shape[::-1])
            elif options.structureList.SMPL.network == 'poseNet':
                self.SMPLdec = poseNet( 
                    self.options.structureList.SMPL.latent_shape, 
                    inShape = self.options.inShape,
                    numiters= 3)    # maybe 1 would be better 
            else:
                raise ValueError(options.structureList.SMPL.network)
            
        # --------- displacements encoder and decoder ---------
        if options.structureList.disps.enable:
            self.dispenc = simpleMLP(
                self.options.structureList.disp.infeature, 
                self.options.structureList.disp.latent_shape,
                self.options.structureList.disp.shape)
            if options.structureList.disp.network == 'simple':
                self.dispdec = simpleMLP(
                    self.options.structureList.disp.latent_shape,
                    self.options.structureList.disp.infeature,
                    self.options.structureList.disp.shape[::-1])
            elif self.options.structureList.disp.network == 'dispGraphNet':
                assert A is not None and ref_vertices is not None, \
                    "Graph Net requires Adjacent matrix and verteices"
                self.dispdec = dispGraphNet(
                    A, ref_vertices, 
                    infeature = self.options.structureList.SMPL.latent_shape,
                    inShape = self.options.inShape)
            else:
                raise ValueError(options.structureList.disp.network)
        
        # --------- texture encoder and decoder ---------
        if options.structureList.text.enable:
            if options.structureList.text.network[:6] == 'simple':    # for vertex
                self.textenc = simpleMLP(
                    self.options.structureList.text.infeature, 
                    self.options.structureList.text.latent_shape,
                    self.options.structureList.text.shape)
            elif options.structureList.text.network[:7] == 'downNet': # for uv map
                self.textenc = downNet()
            else:
                raise ValueError(options.structureList.text.network)
            
            if options.structureList.text.network[-6:] == 'simple':   # for vertex
                self.textdec = simpleMLP(
                    self.options.structureList.text.latent_shape,
                    self.options.structureList.text.infeature, 
                    self.options.structureList.text.shape[::-1])
            elif options.structureList.text.network[-5:] == 'upNet':
                self.textdec = upNet(
                    self.options.structureList.SMPL.latent_shape,
                    inShape = self.options.inShape, 
                    outdim = 3)
            else:
                raise ValueError(options.structureList.text.network)
        
        # --------- camera encoder and decoder ---------
        if options.structureList.camera.enable:
            self.cameraenc = simpleMLP(
                self.options.structureList.camera.infeature, 
                self.options.structureList.camera.latent_shape,
                self.options.structureList.camera.shape)
            if options.structureList.camera.network == 'simple':
                self.cameradec = simpleMLP(
                    self.options.structureList.camera.latent_shape,
                    self.options.structureList.camera.infeature,
                    self.options.structureList.camera.shape[::-1])
            elif options.structureList.camera.network == 'cam':
                self.SMPLdec = cameraNet( 
                    self.options.structureList.SMPL.latent_shape, 
                    inShape = self.options.inShape,
                    numiters= 3)    # maybe 1 would be better 
            else:
                raise ValueError(options.structureList.text.network)
        
        # --------- image encoder and decoder ---------
        self.imageenc = downNet()
        self.imagedec = upNet(
            infeature = 2048, 
            inShape = self.options.inShape, 
            outdim = 3)
        
        
    def forward(self, x, init_cam = None, init_pose = None, init_shape = None):
                
        # extract latent codes 
        codes, connections = self.imageenc(x)
        
        # reconctruct index map
        indexMap = self.imagedec(codes, connections)
        
        
        
        
        out = {'latentCode': codes,
               'indexMap'  : indexMap,
               }
        
        
        return out  
      

options = namedtuple('options', structure_options.keys())(**structure_options)
mesh = Mesh(options, 0, device = 'cpu')

verif = multiple_downstream(options, mesh.adjmat, mesh.ref_vertices.t(), 'cpu')
out = verif(torch.Tensor(1, 3,224,224),
            init_cam = torch.tensor([0.9, 0., 0.]).view(1, 3),
            init_pose = torch.zeros(144)[None,:],
            init_shape = torch.zeros(10)[None, :])