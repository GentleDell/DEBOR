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

import torch.nn as nn

from .structure_loss import latentCode_loss, supervision_loss, rendering_loss
from .subtasks_nn import simpleMLP
from .subtasks_nn import upNet, downNet, dispGraphNet, iterativeNet

class DEBORNet(nn.Module):
    def __init__(self, structure_options, A = None, ref_vertices = None, 
                 avgPose=None, avgBeta=None, avgTrans=None, avgCam=None,
                 device = 'cuda'):
        super(DEBORNet, self).__init__()
        
        self.options = structure_options
        
        self.SMPLcfg = namedtuple(
            'options', 
            self.options.structureList['SMPL'].keys())\
                (**self.options.structureList['SMPL'])
        self.dispcfg = namedtuple(
            'options', 
            self.options.structureList['disp'].keys())\
            (**self.options.structureList['disp'])
        self.textcfg = namedtuple(
            'options', 
            self.options.structureList['text'].keys())\
            (**self.options.structureList['text'])
        self.cameracfg = namedtuple(
            'options', 
            self.options.structureList['camera'].keys())\
            (**self.options.structureList['camera'])
        self.indMapcfg = namedtuple(
            'options', 
            self.options.structureList['indexMap'].keys())\
            (**self.options.structureList['indexMap'])
        self.rendercfg = namedtuple(
            'options', 
            self.options.structureList['rendering'].keys())\
            (**self.options.structureList['rendering'])
        
        # --------- SMPL model encoder and decoder ---------
        if self.SMPLcfg.enable:
            self.SMPLenc = simpleMLP(
                self.SMPLcfg.infeature, 
                self.SMPLcfg.latent_shape,
                layers = self.SMPLcfg.shape)
            if self.SMPLcfg.network == 'simple':
                self.SMPLdec = simpleMLP(
                    self.SMPLcfg.latent_shape,
                    self.SMPLcfg.infeature,
                    layers = self.SMPLcfg.shape[::-1])
                
            elif self.SMPLcfg.network == 'iterNet':
                assert avgBeta is not None and avgPose is not None\
                    and avgTrans is not None and avgCam is not None,\
                        "iterNet requires the average pose and shape."
                self.SMPLdec = iterativeNet( 
                    self.SMPLcfg.latent_shape,
                    initPose = avgPose,
                    initBeta = avgBeta,
                    initTrans= avgTrans,
                    initCam  = avgCam,
                    numiters = 3,
                    withActFunc=self.SMPLcfg.actFunc)
            else:
                raise ValueError(self.SMPLcfg.network)
            
        # --------- displacements encoder and decoder ---------
        if self.dispcfg.enable:
            self.dispenc = simpleMLP(
                self.dispcfg.infeature, 
                self.dispcfg.latent_shape,
                layers = self.dispcfg.shape)
            if self.dispcfg.network == 'simple':
                self.dispdec = simpleMLP(
                    self.dispcfg.latent_shape,
                    self.dispcfg.infeature,
                    layers = self.dispcfg.shape[::-1])
            elif self.dispcfg.network == 'dispGraphNet':
                assert A is not None and ref_vertices is not None, \
                    "Graph Net requires Adjacent matrix and verteices"
                self.dispdec = dispGraphNet(
                    A, ref_vertices, 
                    infeature = self.SMPLcfg.latent_shape,
                    inShape = self.options.inShape)
            else:
                raise ValueError(self.dispcfg.network)
        
        # --------- texture encoder and decoder ---------
        if self.textcfg.enable:
            if self.textcfg.network[:6] == 'simple': 
                # for vertex
                self.textenc = simpleMLP(
                    self.textcfg.infeature, 
                    self.textcfg.latent_shape,
                    layers = self.textcfg.shape)
            elif self.textcfg.network[:7] == 'downNet': 
                # for uv map
                raise NotImplementedError(
                    "might be harmful since uvmap is constant " +
                    "under different body, disp and camera.")
                self.textenc = downNet()
            else:
                raise ValueError(self.textcfg.network)
            
            if self.textcfg.network[-6:] == 'simple':   
                # for vertex
                self.textdec = simpleMLP(
                    self.textcfg.latent_shape,
                    self.textcfg.infeature, 
                    layers = self.textcfg.shape[::-1])
            elif self.textcfg.network[-5:] == 'upNet':
                # for uv map
                raise NotImplementedError(
                    "might be harmful since uvmap is constant " +
                    "under different body, disp and camera.")
                self.textenc = downNet()    # force the encoder to be downNet
                self.textdec = upNet(
                    self.SMPLcfg.latent_shape,
                    inShape = self.options.inShape, 
                    outdim = 3)
            else:
                raise ValueError(self.textcfg.network)
        
        # --------- camera encoder and decoder ---------
        if self.cameracfg.enable and (self.SMPLcfg.network != 'iterNet'):
            self.cameraenc = simpleMLP(
                self.cameracfg.infeature, 
                self.cameracfg.latent_shape,
                layers = self.cameracfg.shape)
            self.cameradec = simpleMLP(
                self.cameracfg.latent_shape,
                self.cameracfg.infeature,
                layers = self.cameracfg.shape[::-1])
        
        # --------- image encoder and decoder ---------
        # In tex2Shape, unet uses LeakyReLU in downsampling while use ReLU in 
        # upsampling. This might not affect the result too much.
        self.imageenc = downNet()
        self.imagedec = upNet(infeature = 2048, outdim = 3)
        
        # --------- loss for the structure ---------
        self.latent_loss = latentCode_loss(structure_options)
        self.supVis_loss = supervision_loss(structure_options)
        if self.rendercfg.enable:
            self.render_loss = rendering_loss(structure_options)
        
        # TODO
        # self.interBatch_loss Try it later
        
    def computeLoss(self, pred, latentCode, GT):
        latentCodeLoss = self.latent_loss(latentCode)
        superviseLoss  = self.supVis_loss(pred, GT)
        
        latCodeLossSum = \
            self.SMPLcfg.weight['latentCode']\
                *latentCodeLoss['smplCode_loss'] +\
            self.dispcfg.weight['latentCode']\
                *latentCodeLoss['dispCode_loss'] +\
            self.textcfg.weight['latentCode']\
                *latentCodeLoss['textCode_loss'] +\
            self.cameracfg.weight['latentCode']\
                *latentCodeLoss['cameraCode_loss']
        supVisLossSum = \
            self.indMapcfg.weight\
                *superviseLoss['indexMapSupv_loss'] +\
            self.SMPLcfg.weight['supervision']\
                *superviseLoss['smplSupv_loss'] +\
            self.dispcfg.weight['supervision']\
                *superviseLoss['dispSupv_loss'] +\
            self.textcfg.weight['supervision']\
                *superviseLoss['textSupv_loss'] +\
            self.cameracfg.weight['supervision']\
                *superviseLoss['cameraSupv_loss']
        
        if self.rendercfg.enable:
            renderingLoss  = self.render_loss(GT['img'], pred, GT)
            renderLossSum = \
                self.rendercfg.weight['proj_weight']\
                    *renderingLoss['projLoss'] +\
                self.rendercfg.weight['render_weight']\
                    *renderingLoss['textLoss']
        else:
            renderLossSum, renderingLoss = 0, 0
            
        outLoss = {'loss': supVisLossSum + latCodeLossSum + renderLossSum,
                   'supVisLoss': supVisLossSum,
                   'latCodeLoss': latCodeLossSum,
                   'renderLoss': renderingLoss}
        return outLoss
        
    def forward(self, image, GT, init_cam = None, init_pose = None, 
                init_shape = None):
        # extract latent codes 
        codes, connections, aggCode = self.imageenc(image)
        
        # reconctruct index map
        indexMap = self.imagedec(codes, connections)
 
        prediction = {'indexMap'  : indexMap}
        latentCode = {'imgAggCode': aggCode}   
 
        if self.SMPLcfg.enable:
            assert GT["SMPL"] is not None, \
                "to train SMPL branch, SMPLParams must be given."
            SMPL_encode = self.SMPLenc(GT["SMPL"])
            SMPL_decode = self.SMPLdec(
                aggCode[:,self.SMPLcfg.latent_start:
                    self.SMPLcfg.latent_start + self.SMPLcfg.latent_shape])
            latentCode['SMPL'] = SMPL_encode
            prediction['SMPL'] = SMPL_decode
        
        if self.dispcfg.enable:
            assert GT["disp"] is not None, \
                "to train disp branch, GT disp must be given."
            disp_encode = self.dispenc(GT["disp"])
            disp_decode = self.dispdec(
                aggCode[:,self.dispcfg.latent_start:
                    self.dispcfg.latent_start + self.dispcfg.latent_shape])
            latentCode['disp'] = disp_encode
            prediction['disp'] = disp_decode
        
        if self.textcfg.enable:
            raise ValueError('modify the structure options of latent code')
            assert GT["text"] is not None, \
                "to train texture branch, GT texture/color must be given."
            text_encode = self.textenc(GT["text"])
            text_decode = self.textdec(
                aggCode[:,self.textcfg.latent_start:
                    self.textcfg.latent_start + self.textcfg.latent_shape])
            latentCode['text'] = text_encode
            prediction['text'] = text_decode
        
        if self.cameracfg.enable and (self.SMPLcfg.network != 'iterNet'):
            assert GT["camera"] is not None, \
                "to train camera branch, GT camera must be given."
            camera_encode = self.cameraenc(
                GT["camera"]['f_rot'].squeeze(dim = 1).float())
            camera_decode = self.cameradec(
                aggCode[:,self.cameracfg.latent_start:
                    self.cameracfg.latent_start+self.cameracfg.latent_shape])
            latentCode['camera'] = camera_encode
            prediction['camera'] = camera_decode
        
        return prediction, latentCode
