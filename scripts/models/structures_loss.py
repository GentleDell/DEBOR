#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 23:13:35 2020

@author: zhantao
"""
import random
from os.path import join as pjn

import torch 
import torch.nn as nn

import camera
import SMPL
from utils.vis_util import read_Obj

class latentCode_loss(nn.Module):
    '''
    Latent codes loss is to constrain the learned low dimension latent 
    representation to contain the information of the target field, such as
    pose, shape, displacements. 
    
    Plane Unet and Graph net can easily overfit to our dataset while they
    still underfit the testing set. This could indicate that the network 
    does not learn things/distructions correctly. So the latent code loss 
    is added.
    
    These losses are defined as:
        loss = || Enc(Img)[i:j,:] - Enc_x(X) ||, x can be smpl, disp, etc.

    According to the structures, this class computes the loss for 
    SMPLCode, dispCode, textCode, cameraCode and returns a dict of
    
    {'smplCode_loss', 'dispCode_loss', 'textCode_loss', 'cameraCode_loss'}
        
    '''
    def __init__(self, options):
        super(latentCode_loss, self).__init__()
        
        self.options = options
        self.SMPLcfg = options.structureList.SMPL
        self.dispcfg = options.structureList.disp
        self.textcfg = options.structureList.text
        self.cameracfg = options.structureList.camera
            
        self.SMPLlossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.SMPLcfg.latent_lossFunc == 'L2']
        self.displossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.dispcfg.latent_lossFunc == 'L2']
        self.textlossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.textcfg.latent_lossFunc == 'L2']
        self.cameralossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.cameracfg.latent_lossFunc == 'L2']
        
    def forward(self, imgCode, SMPLCode = None, dispCode = None, 
                textCode = None, cameraCode = None):
        latCodeLoss = {'smplCode_loss': 0,
                       'dispCode_loss': 0,
                       'textCode_loss': 0,
                       'cameraCode_loss' : 0}
        
        if self.SMPLcfg.enable:
            assert SMPLCode is not None                
            target = imgCode[:,
                             self.SMPLcfg.latent_start:
                             self.SMPLcfg.latent_shape]
            latCodeLoss['smplCode_loss'] = self.SMPLlossfunc(SMPLCode, target)
        if self.dispcfg.enable:
            assert dispCode is not None
            target = imgCode[:,
                             self.dispcfg.latent_start:
                             self.dispcfg.latent_shape]
            latCodeLoss['dispCode_loss'] = self.displossfunc(dispCode, target)
        if self.textcfg.enable:
            assert textCode is not None
            target = imgCode[:,
                             self.textcfg.latent_start:
                             self.textcfg.latent_shape]
            latCodeLoss['textCode_loss'] = self.textlossfunc(textCode, target)
        if self.cameracfg.enable:
            assert cameraCode is not None
            target = imgCode[:,
                             self.cameracfg.latent_start:
                             self.cameracfg.latent_shape]
            latCodeLoss['cameraCode_loss'] = \
                self.cameralossfunc(cameraCode, target)
        
        return latCodeLoss
        
class supervision_loss(nn.Module):
    '''
    Supervision loss supervises networks to learn the target information 
    explicitly. 
    
    The loss is defined as:
        loss = || gt_x - Dec_x(Enc(img)[i:j,:]) ||, x can be smpl, disp, etc.
    
    Returns a dict of:
    
    {'smplSupv_loss', 'dispSupv_loss', 'textSupv_loss', 'cameraSupv_loss',
     'indexMapSupv_loss'}
        
    '''
    def __init__(self, options):
        super(supervision_loss, self).__init__()
        self.options = options
        self.SMPLcfg = options.structureList.SMPL
        self.dispcfg = options.structureList.disp
        self.textcfg = options.structureList.text
        self.cameracfg = options.structureList.camera
        self.indMapcfg = options.structureList.indexMap
        
        self.SMPLlossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.SMPLcfg.supVis_lossFunc == 'L2']
        self.displossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.dispcfg.supVis_lossFunc == 'L2']
        self.textlossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.textcfg.supVis_lossFunc == 'L2']
        self.cameralossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.cameracfg.supVis_lossFunc == 'L2']
        self.indMapLossfunc = (nn.L1Loss(), nn.L2Loss())\
                [self.indMapcfg.supVis_lossFunc == 'L2']
        
    def forward(self, prediction, imageGT, SMPLGT = None, 
                dispGT = None, textGT = None, cameraGT = None):
        supVisLoss = {'smplSupv_loss': 0,
                      'dispSupv_loss': 0,
                      'textSupv_loss': 0,
                      'cameraSupv_loss': 0,
                      'indexMapSupv_loss': 0}
        
        if self.SMPLcfg.enable:
            assert SMPLGT is not None
            supVisLoss['smplSupv_loss'] = \
                self.SMPLlossfunc(prediction['SMPL'], SMPLGT)
        if self.dispcfg.enable:
            assert dispGT is not None
            supVisLoss['dispSupv_loss'] = \
                self.displossfunc(prediction['disp'], dispGT)
        if self.textcfg.enable:
            assert textGT is not None
            supVisLoss['textSupv_loss'] = \
                self.textlossfunc(prediction['text'], textGT)
        if self.cameracfg.enable:
            assert cameraGT is not None
            supVisLoss['cameraSupv_loss'] = \
                self.cameralossfunc(prediction['camera'], cameraGT)
        supVisLoss['indexMapSupv_loss'] = \
            self.indMapLossfunc(prediction['indexMap'], imageGT)
        
        return supVisLoss
        
class rendering_loss(nn.Module):
    '''
    To make sure that the predicted parameters are realistic and reasonable, 
    the rendering loss is added.
    
    Currently, only the projection loss is implemented.
    '''
    def __init__(self, options):
        super(self.indexMapSupv_loss, self).__init__()
        self.options = options
        self.rendercfg = options.structureList.renderings
        
        self.projLossFunc = \
            (nn.L1Loss(), nn.L2Loss())[self.rendercfg.proj_lossFunc == 'L2']
        self.renderLossFunc = \
            (nn.L1Loss(), nn.L2Loss())[self.rendercfg.render_lossFunc == 'L2']
        
        # prepare data for projection
        _, self.faces, _, _ = read_Obj(options.smpl_objfile_path)
        self.faces = torch.tensor(self.faces)[None]
        self.SMPLmodel = SMPL(options.smpl_model_path, 'cuda')
        self.persCamera= camera()
        self.numSamples= 6890*self.rendercfg.sampleRate
        
    def forward(self, image, predictions, GT):

        batchSize= image.shape[0]
        SMPLPred = predictions['SMPL']
        dispPred = predictions['disp']
        cameraPred = predictions['camera']
        
        raise NotImplementedError('change to 6d representation')
        predBody = self.SMPLmodel(
            SMPLPred[:, :72], 
            SMPLPred[:, 72:82], 
            SMPLPred[:, 82:85])
        
        predPixels, _ = self.persCamera(
            fx=cameraPred[:,0], 
            fy=cameraPred[:,1],  
            cx=cameraPred[:,2], 
            cy=cameraPred[:,3], 
            rotation=cameraPred[:,4:13], 
            translation=cameraPred[:,13:], 
            points=predBody + dispPred, 
            faces=self.faces.repeat_interleave(batchSize, dim = 0), 
            visibilityOn = False)
        
        raise NotImplementedError('change to 6d representation')
        GTBody = self.SMPLmodel(GT['SMPL'][:, :72], 
                                GT['SMPL'][:, 72:82], 
                                GT['SMPL'][:, 82:85])
        GTPixels, _ = self.persCamera(
            fx=GT['camera'][:,0], 
            fy=GT['camera'][:,1],  
            cx=GT['camera'][:,2], 
            cy=GT['camera'][:,3], 
            rotation=GT['camera'][:,4:13], 
            translation=GT['camera'][:,13:], 
            points=GTBody+ GT['disp'], 
            faces=self.faces.repeat_interleave(batchSize, dim = 0), 
            visibilityOn = False)
        
        indice = random.sample(range(6890), self.numSamples)
        sampleInd = torch.tensor(indice)
        
        # use textloss after roughly converge    
        renderLoss = {'projLoss': 0,
                      'textLoss': 0}        
        for cnt in range(batchSize):        
            renderLoss['projLoss'] = \
                renderLoss['projLoss'] + \
                    self.projLossFunc(
                        predPixels[:, sampleInd],
                        GTPixels[:, sampleInd])
        
        return renderLoss
        
class batch_loss(nn.Module):
    '''
    This loss function tries to leverage training samples of the same batch.
    
    If two samples are from the same object, then they should have similar
    dispCode, textCode, smplCode and corresponding predictions. But since we 
    augment input images, the batch_loss_similar does not apply to cameraCode
    and prediction as well as the indexMap prediction.
    '''
    def __init__(self, options):
        super(batch_loss, self).__init__()
        self.options = options
        print('batch loss currently only support L2 loss')
        print('only use batch loss after roughly converging!')
        self.lossfunc = nn.L2Loss()
        
    def forward(self, inputNames, predictions, latentCodes):
        
        batchLoss = {'smplBatchLoss_similarCode': 0,
                     'dispBatchLoss_similarCode': 0,
                     'textBatchLoss_similarCode': 0,
                     'smplBatchLoss_similarPred': 0,
                     'dispBatchLoss_similarPred': 0,
                     'textBatchLoss_similarPred': 0,
                     }
        
        for srcInd in range(len(inputNames)-1):
            for dstInd in range(1, len(inputNames)):
                
                # +1 means similar, -1 means dissimilar
                sign = (-1,+1)[inputNames['srcInd'] == inputNames['dstInd']]
    
                # ---------- SMPL ----------
                batchLoss['smplBatchLoss_similarCode'] = \
                    batchLoss['smplBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['SMPL'][srcInd],
                            latentCodes['SMPL'][dstInd])
                
                batchLoss['smplBatchLoss_similarPred'] = \
                    batchLoss['smplBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['SMPL'][srcInd],
                            predictions['SMPL'][dstInd])
                    
                # ---------- disp ----------
                batchLoss['dispBatchLoss_similarCode'] = \
                    batchLoss['dispBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['disp'][srcInd],
                            latentCodes['disp'][dstInd])
                
                batchLoss['dispBatchLoss_similarPred'] = \
                    batchLoss['dispBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['disp'][srcInd],
                            predictions['disp'][dstInd])
                        
                # ---------- text ----------
                batchLoss['textBatchLoss_similarCode'] = \
                    batchLoss['textBatchLoss_similarCode'] + \
                        sign*self.lossfunc(
                            latentCodes['text'][srcInd],
                            latentCodes['text'][dstInd])
                
                batchLoss['textBatchLoss_similarPred'] = \
                    batchLoss['textBatchLoss_similarPred'] + \
                        sign*self.lossfunc(
                            predictions['text'][srcInd],
                            predictions['text'][dstInd])    
                
                # ---------- camera ----------
                # not support camera and indexMap yet
        
        return batchLoss