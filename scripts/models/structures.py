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

import torch.nn as nn

from structure_loss import latentCode_loss, supervision_loss, rendering_loss
from subtasks_nn import simpleMLP, cameraNet
from subtasks_nn import poseNet, upNet, downNet, dispGraphNet

class DEBORNet(nn.Module):
    def __init__(self, structure_options, A = None, ref_vertices = None, 
                 device = 'cuda'):
        super(DEBORNet, self).__init__()
        
        self.options = structure_options
        self.SMPLcfg = self.options.structureList.SMPL
        self.dispcfg = self.options.structureList.disp
        self.textcfg = self.options.structureList.text
        self.cameracfg = self.options.structureList.camera
        self.indMapcfg = self.options.structureList.indexMap
        
        # --------- SMPL model encoder and decoder ---------
        if self.SMPLcfg.enable:
            self.SMPLenc = simpleMLP(
                self.SMPLcfg.infeature, 
                self.SMPLcfg.latent_shape,
                self.SMPLcfg.shape)
            if self.SMPLcfg.network == 'simple':
                self.SMPLdec = simpleMLP(
                    self.SMPLcfg.latent_shape,
                    self.SMPLcfg.infeature,
                    self.SMPLcfg.shape[::-1])
            elif self.SMPLcfg.network == 'poseNet':
                self.SMPLdec = poseNet( 
                    self.SMPLcfg.latent_shape, 
                    inShape = self.options.inShape,
                    numiters= 3)    # maybe 1 would be better 
            else:
                raise ValueError(self.SMPLcfg.network)
            
        # --------- displacements encoder and decoder ---------
        if self.options.structureList.disps.enable:
            self.dispenc = simpleMLP(
                self.dispcfg.infeature, 
                self.dispcfg.latent_shape,
                self.dispcfg.shape)
            if self.dispcfg.network == 'simple':
                self.dispdec = simpleMLP(
                    self.dispcfg.latent_shape,
                    self.dispcfg.infeature,
                    self.dispcfg.shape[::-1])
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
                    self.textcfg.shape)
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
                    self.textcfg.shape[::-1])
            elif self.textcfg.network[-5:] == 'upNet':
                # for uv map
                raise NotImplementedError(
                    "might be harmful since uvmap is constant " +
                    "under different body, disp and camera.")
                self.textenc = downNet()
                self.textdec = upNet(
                    self.SMPLcfg.latent_shape,
                    inShape = self.options.inShape, 
                    outdim = 3)
            else:
                raise ValueError(self.textcfg.network)
        
        # --------- camera encoder and decoder ---------
        if self.cameracfg.enable:
            self.cameraenc = simpleMLP(
                self.cameracfg.infeature, 
                self.cameracfg.latent_shape,
                self.cameracfg.shape)
            if self.cameracfg.network == 'simple':
                self.cameradec = simpleMLP(
                    self.cameracfg.latent_shape,
                    self.cameracfg.infeature,
                    self.cameracfg.shape[::-1])
            elif self.cameracfg.network == 'cam':
                self.SMPLdec = cameraNet( 
                    self.SMPLcfg.latent_shape, 
                    inShape = self.options.inShape,
                    numiters= 3)    # maybe 1 would be better 
            else:
                raise ValueError(self.textcfg.network)
        
        # --------- image encoder and decoder ---------
        self.imageenc = downNet()
        self.imagedec = upNet(
            infeature = 2048, 
            inShape = self.options.inShape, 
            outdim = 3)
        
        # --------- loss for the structure ---------
        self.latent_loss = latentCode_loss(structure_options)
        self.supVis_loss = supervision_loss(structure_options)
        self.render_loss = rendering_loss(structure_options)
        
        # TODO
        # self.interBatch_loss Try it later
        
    def computeLoss(self, inputImg, pred, GT, latentCode):
        latentCodeLoss = self.latent_loss(
            latentCode['imageCode'],
            latentCode['SMPLCode'], latentCode['dispCode'], 
            latentCode['textCode'], latentCode['cameraCode'])
        superviseLoss  = self.supVis_loss(
            pred, GT['imageGT'], GT['SMPLGT'], GT['dispGT'], 
            GT['textGT'], GT['cameraGT'])
        renderingLoss  = self.render_loss(inputImg, pred, GT)
        
        latCodeLossSum = \
            self.SMPLcfg.weight.latentCode*latentCodeLoss['smplCode_loss'] +\
            self.dispcfg.weight.latentCode*latentCodeLoss['dispCode_loss'] +\
            self.textcfg.weight.latentCode*latentCodeLoss['textCode_loss'] +\
            self.cameracfg.weight.latentCode*latentCodeLoss['cameraCode_loss']
        supVisLossSum = \
            self.indMapcfg.weight*superviseLoss['indexMapSupv_loss'] +\
            self.SMPLcfg.weight.supervision*superviseLoss['smplSupv_loss'] +\
            self.dispcfg.weight.supervision*superviseLoss['dispSupv_loss'] +\
            self.textcfg.weight.supervision*superviseLoss['textSupv_loss'] +\
            self.cameracfg.weight.supervision*superviseLoss['cameraSupv_loss']
            
        outLoss = {'loss': supVisLossSum + latCodeLossSum + renderingLoss,
                   'supVisLoss': supVisLossSum,
                   'latCodeLoss': latCodeLossSum,
                   'renderLoss': renderingLoss}
        return outLoss
        
    def forward(self, image, SMPLParams = None, disps = None, text = None,
                camera = None, init_cam = None, init_pose = None, 
                init_shape = None):
                
        # extract latent codes 
        codes, connections, aggCode = self.imageenc(image)
        
        # reconctruct index map
        indexMap = self.imagedec(codes, connections)
 
        prediction = {'indexMap'  : indexMap}
        latentCode = {'latentCode': codes}   
 
        if self.SMPLcfg.enable:
            assert SMPLParams is not None, \
                "to train SMPL branch, SMPLParams must be given."
            SMPL_encode = self.SMPLenc(SMPLParams)
            SMPL_decode = self.SMPLdec(
                aggCode[:,self.SMPLcfg.latent_start:
                          self.SMPLcfg.latent_shape])
            latentCode['SMPL'] = SMPL_encode
            prediction['SMPL'] = SMPL_decode
        
        if self.dispcfg.enable:
            assert disps is not None, \
                "to train disp branch, GT disp must be given."
            disp_encode = self.dispenc(disps)
            disp_decode = self.dispdec(
                aggCode[:,self.dispcfg.latent_start:
                          self.dispcfg.latent_shape])
            latentCode['disp'] = disp_encode
            prediction['disp'] = disp_decode
        
        if self.textcfg.enable:
            assert text is not None, \
                "to train texture branch, GT texture/color must be given."
            text_encode = self.textenc(text)
            text_decode = self.textdec(
                aggCode[:,self.textcfg.latent_start:
                          self.textcfg.latent_shape])
            latentCode['text'] = text_encode
            prediction['text'] = text_decode
        
        if self.cameracfg.enable:
            assert camera is not None, \
                "to train camera branch, GT camera must be given."
            camera_encode = self.cameraenc(camera)
            camera_decode = self.cameradec(
                aggCode[:,self.cameracfg.latent_start:
                          self.cameracfg.latent_shape])
            latentCode['camera'] = camera_encode
            prediction['camera'] = camera_decode
        
        return prediction, latentCode
