#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:19:32 2021

@author: Zhantao
"""

import argparse
from os.path import abspath, isfile, join as pjn
import sys 
if abspath('../') not in sys.path:
    sys.path.append(abspath('../'))
if abspath('./MGN_helper') not in sys.path:
    sys.path.append(abspath('./MGN_helper'))
if abspath('./MGN_helper/lib') not in sys.path:
    sys.path.append(abspath('./MGN_helper/lib'))
if abspath('./MGN_helper/utils') not in sys.path:
    sys.path.append(abspath('./MGN_helper/utils'))
import pickle as pkl
from glob import glob

import yaml
import torch

import numpy as np
import scipy
from psbody.mesh import Mesh, MeshViewers

from MGN_helper.utils.smpl_paths import SmplPaths
from ch_smpl import Smpl
from helper_dataset import smplFromParas, compute_offset_tPose, check_folder
from helper_dataset import seg_filter, create_subject


def save_offsets(offsets_hres, offsets_std, savePath):
    """save the given offsets to the given path."""
    check_folder(savePath)
    
    with open(pjn(savePath,'offsets_hres.npy'), 'wb') as f:
        np.save(f, offsets_hres)             
    with open(pjn(savePath, 'offsets_std.npy'), 'wb') as f:
        np.save(f, offsets_std)       

class MGN_bodyAug_preparation(object):
    '''
    This class enriches the MGN main dataset with the poses and garments in the
    MGN wardrobe dataset. 
    
    At frist, for each subject in the MGN main dataset, we compute its offsets
    by subtracting the naked mesh from the registered body mesh which is given 
    in the dataset.
    
    Then, for each subject, we randomly pick a suit with the same style as the 
    suit of the subject from the wardrobe and randomly sample a body from both 
    the main and wardrobe dataset. With the body and suit, we thus can generate
    new subjects and compute the corresponding offsets. The texture of the new
    subject is borrowed from the original subject, as wardrobe does not provide
    texture information.
    
    We use normal guided method to compute the offsets given body and suit, as 
    some provided meshs of suit are not good enough, as explained in my slide,
    which would bring defects to the generated subject if using the 'vert_ind'
    directly. Besides, in standard resolution (6890), the offsets computed by 
    NGM is more smooth then the one downsampled from the hres one. 
    
    All offsets generated by this class are in t-pose/0-pose, under the SMPL
    coordinate system. The meshes of garments are in t-pose with trans being
    [0,0,0] under the SMPL coordiante system (same as original MGN dataset).
    So, do not forget to convert the corrd sys when use the offsets and the gt 
    garment meshes.
    '''
    def __init__(self, cfg):
                
        self.path_smpl = cfg['smplModel']
        self.path_subjects = sorted( glob(pjn(cfg['datarootMGN'], '*')) )
        self.path_wardrobe = sorted( glob(pjn(cfg['wardrobeMGN'], '*')) )
        
        ## <==== collect subjects and the corresponding garments
        self.text, self.disp = [], []
        self.betas, self.poses, self.trans = [], [], []
        self.subjectSuit = []
        for subjectPath in self.path_subjects:
            # collect smpl parameters of the subjects
            pathRegistr  = pjn(subjectPath, 'registration.pkl')
            registration = pkl.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( torch.Tensor(registration['betas'][None,:]) )
            self.poses.append( torch.Tensor(registration['pose'][None,:]) )
            self.trans.append( torch.Tensor(registration['trans'][None,:]) )

            # we need the original garment style as we have to use its texture for augmentation
            suit = []
            if isfile(pjn(subjectPath, 'Pants.obj')): suit.append('Pants')
            if isfile(pjn(subjectPath, 'ShortPants.obj')): suit.append('ShortPants')
            if isfile(pjn(subjectPath, 'ShirtNoCoat.obj')): suit.append('ShirtNoCoat')
            if isfile(pjn(subjectPath, 'TShirtNoCoat.obj')): suit.append('TShirtNoCoat')
            if isfile(pjn(subjectPath, 'LongCoat.obj')): suit.append('LongCoat')
            self.subjectSuit.append(suit)
        self.MGNSize_main = len(self.betas)
            
        ## <==== collect additional garments and body parameters
        self.wardrobe = {
            'Pants': [],  'Pants_ind': [],
            'ShortPants': [],  'ShortPants_ind': [],
            'ShirtNoCoat': [],  'ShirtNoCoat_ind': [],
            'TShirtNoCoat': [],  'TShirtNoCoat_ind': [], 
            'LongCoat': [],  'LongCoat_ind': []
            }
        # inproper garments:
        #   1. tightly-coupled with pose
        #   2. having strange/incorrect structure, e.g. incorrect segmentation 
        #   3. involve irrelevant parts as garments, e.g. hair
        self.skipSamples = ['005', '006', '017', '030', '038', '041', '051',   # fixed: 2,7,8,12,23,25,28,35,54,76,99
                            '059', '084', '085']                               # skipped: 5,6,17,30(no corr. tex),38,41,51,59,85
                        
        for garments in self.path_wardrobe:
            # collect smpl parameters in wardrobe to enrich the dataset
            pathRegistr  = pjn(garments, 'registration.pkl')
            registration = pkl.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( torch.Tensor(registration['betas'][None,:]) )
            self.poses.append( torch.Tensor(registration['pose'][None,:]) )
            self.trans.append( torch.Tensor(registration['trans'][None,:]) )
            
            if garments[-3:] in self.skipSamples:
                continue    # skip inproper garments samples
            
            suit = '  '.join(glob(pjn(garments, '*.obj')))
            if 'ShortPants' in suit:
                self.wardrobe['ShortPants'].append(garments)
                self.wardrobe['ShortPants_ind'].append(garments[-3:])
            elif 'Pants' in suit:
                self.wardrobe['Pants'].append(garments)
                self.wardrobe['Pants_ind'].append(garments[-3:])
            if 'TShirtNoCoat' in suit:
                self.wardrobe['TShirtNoCoat'].append(garments)
                self.wardrobe['TShirtNoCoat_ind'].append(garments[-3:])
            elif 'ShirtNoCoat' in suit:
                self.wardrobe['ShirtNoCoat'].append(garments)
                self.wardrobe['ShirtNoCoat_ind'].append(garments[-3:])
            elif 'LongCoat' in suit:
                self.wardrobe['LongCoat'].append(garments)
                self.wardrobe['LongCoat_ind'].append(garments[-3:])
        # convert to np array for fast access
        self.wardrobe['ShortPants_ind'] = np.array(self.wardrobe['ShortPants_ind'])
        self.wardrobe['Pants_ind'] = np.array(self.wardrobe['Pants_ind'])
        self.wardrobe['TShirtNoCoat_ind'] = np.array(self.wardrobe['TShirtNoCoat_ind'])
        self.wardrobe['ShirtNoCoat_ind'] = np.array(self.wardrobe['ShirtNoCoat_ind'])
        self.wardrobe['LongCoat_ind'] = np.array(self.wardrobe['LongCoat_ind'])
        
        self.MGNSize_wardrobe = len(self.betas) - self.MGNSize_main
        
        ## <==== prepare high resolution SMPL, standard SMPL and downsampling mat
        dp = SmplPaths()
        self.hresSMPL = Smpl(dp.get_hres_smpl_model_data())
        self.stdSMPL  = Smpl(dp.get_smpl_file())
        self.downMat  = scipy.sparse.load_npz(cfg['path_downMat'])      # using sparse matrix saves a lot mem while resErr is @1e-6
        self.stdSMPLmesh = Mesh(filename=cfg['smplMesh_std'])           # std res mesh of SMPL model
        
        ## <==== get configurations for post processing
        self.verbose_on = cfg['show_intermediateMeshes']
        self.ind_start  = cfg['start_from_ind']
        self.save_offsets = cfg['save_displacements']
        self.num_separation = {'hres': cfg['number_separation_hres'],
                               'std' : cfg['number_separation_std' ]}
        self.thresholds = {'coats': {}, 'pants':{}}
        for ind, key in enumerate(sorted(self.thresholds.keys())):
            self.thresholds[key]['max_offsets'] = cfg['max_offsets'][ind]
            self.thresholds[key]['offsets_check_threshold'] = cfg['offsets_check_threshold'][ind]
            self.thresholds[key]['max_diffOffsets'] = cfg['max_diffOffsets'][ind]
    
    def setSuit(self, subInd: int, each_Nsuit: int, poseInd: int):
        '''
        This function randomly chooses n ('each_Nsuit') suits from the wardrobe 
        and make sure the chosen suit have the same style as original subject 
        ('subInd'), as we need to use the texture of the original subject for 
        rendering.  
        
        Since some garments in wardobe are closely coupled with pose, we also 
        need to consider pose. The 'poseInd' is the index of the trgt pose to
        generate new subject. If the coreresponding suit of the trgt pose has 
        the same style as the org sub, we will return the suit.  

        Parameters
        ----------
        subInd : int
            The index to the original subject.
        each_Nsuit : int
            The number of suits to dress each new subject.
        poseInd : int
            The index to the pose to generate new subjects.

        Returns
        -------
        coatPath : List
            List of path to the coats to dress the new body.
        pantsPath : List
            List of path to the pants to dress the new body.

        '''
        
        ## get the original garment style
        origSuit = self.subjectSuit[subInd]
        
        ## randomly choose pants
        pantsInd = np.random.randint(len(self.wardrobe[origSuit[0]]), size = each_Nsuit).tolist()
        pantsPath= [pjn(self.wardrobe[origSuit[0]][ind], origSuit[0]+'.obj') for ind in pantsInd]
        
        ## randomly choose coats for the body 
        if 'Coat' not in '*'.join(origSuit):
            # if a subject does not have coats, from what we observe, it 
            # probably wears a tshir, except for the 83rd which is naked.
            if subInd == 83:
                coatPath = [None]*each_Nsuit    # naked
            else:
                coatInd  = np.random.randint(len(self.wardrobe['TShirtNoCoat']), size = each_Nsuit).tolist()
                coatPath = [pjn(self.wardrobe['TShirtNoCoat'][ind], 'TShirtNoCoat.obj') for ind in coatInd]
        else:
            # pick a coat in the same style of the orignal coat
            coatInd  = np.random.randint(len(self.wardrobe[origSuit[1]]), size = each_Nsuit).tolist()
            coatPath = [pjn(self.wardrobe[origSuit[1]][ind], origSuit[1]+'.obj') for ind in coatInd]
        
        ## check if the pose is from MGN wardrobe. If so, see if it has the 
        ## same style of suit as source subject. If so, add the suit to return. 
        if poseInd >= self.MGNSize_main:                                           # if in wardrobe
            garments_ind_str = '{:0>3d}'.format(poseInd - len(self.path_subjects))
            if garments_ind_str not in self.skipSamples:                           # if not dropped
                pantsInd = np.where(self.wardrobe[origSuit[0]+'_ind'] == garments_ind_str)[0]
                if pantsInd.shape[0] != 0:                                         # if same style
                    pantsPath[0] = pjn(self.wardrobe[origSuit[0]][pantsInd.item()], origSuit[0]+'.obj')
                
                if len(origSuit) == 2:                                             # if has coat
                    coatInd  = np.where(self.wardrobe[origSuit[1]+'_ind'] == garments_ind_str)[0]
                    if coatInd.shape[0] != 0:                                      # if same style
                        coatPath[0] = pjn(self.wardrobe[origSuit[1]][coatInd.item()], origSuit[1]+'.obj') 
        
        return coatPath, pantsPath
    
    def computeOffsets_guided(
            self, coatPath: str, pantsPath: str, tarPara: list, 
            subObj: Mesh = None, subTex: str = None, is_hres: bool = True):
        """use the N G M to compute the offset in t pose."""
        
        smpl = self.hresSMPL.copy() if is_hres else self.stdSMPL.copy()
        splt = self.num_separation[ 'hres' if is_hres else 'std' ]
        
        ## per-vertex offsets
        v_offsets_t = np.zeros_like(smpl.r)
        
        ## Pants
        offset_pants_t = compute_offset_tPose(
            smpl, pantsPath, self.thresholds['pants'], splt, self.verbose_on
            )
        mask = np.linalg.norm(offset_pants_t, axis=1) > np.linalg.norm(v_offsets_t, axis=1)
        v_offsets_t[mask] = offset_pants_t[mask]
        
        ## coat
        # None for sub84 and others subs without coat in their folder
        if coatPath is not None:    
            offset_coat_t = compute_offset_tPose(
                smpl, coatPath, self.thresholds['coats'], splt, self.verbose_on
                )
            mask = np.linalg.norm(offset_coat_t, axis=1) > np.linalg.norm(v_offsets_t, axis=1)
            v_offsets_t[mask] = offset_coat_t[mask]
        
        ## Dress body
        if self.verbose_on and subObj is not None:
            print('show mesh in self.computeOffsets_guided().')
            smpl = smplFromParas(smpl, v_offsets_t, tarPara[0], tarPara[1], tarPara[2])
            dressed_body = Mesh(smpl.r, smpl.f)
            dressed_body.vt = subObj.vt
            dressed_body.ft = subObj.ft
            dressed_body.set_texture_image(subTex)
            
            mvs = MeshViewers((1, 1))
            mvs[0][0].set_static_meshes([dressed_body])
        
        return v_offsets_t
    
    def computeOffsets_direct(self, subObj, tarPara):
        """subtract naked mesh from the registered mesh to compute offsets."""
        ## only works when the registered subject is available
       
        smpl = self.hresSMPL.copy()
        
        ## compute the posed offsets
        offsets = np.zeros_like(smpl.r)
        smpl = smplFromParas(smpl, offsets, tarPara[0], tarPara[1], tarPara[2])
        offsets_p = subObj.v - smpl.r
        
        ## unpose the offsets
        # 1. put on the offsets to body and do inverse pose
        smpl = smplFromParas(smpl, offsets_p, -tarPara[0], tarPara[1], tarPara[2])
        invposed_body_off = smpl.r
        
        # 2. generate naked body in inverse pose
        offsets = np.zeros_like(smpl.r)
        smpl = smplFromParas(smpl, offsets, -tarPara[0], tarPara[1], tarPara[2])
        invposed_body = smpl.r
        
        # 3. get the unposed/t-posed offsets
        offsets_t = invposed_body_off - invposed_body
        
        return offsets_t    
                      
    def generateDataset(self, keepN_pose = 30, each_Npose = 2, each_Nsuit = 2):
        """main function to generate the enriched dataset."""
        
        ## choose additional poses for all subjects in MGN
        # here we choose the probability p such that the expected appearing 
        # times of all poses (196 in total) to be equal (Att: poses in MGN main 
        # appear at least once).
        p = (0.96*each_Npose-1)/1.96/each_Npose    
        p_MGN = p/self.MGNSize_main*np.ones((self.MGNSize_main,))
        p_wdb = (1-p)/(self.MGNSize_wardrobe)*np.ones((self.MGNSize_wardrobe,))
        augPoseInd = np.random.choice(
            np.arange(len(self.poses)), 
            size=(len(self.path_subjects), each_Npose),
            p=np.hstack([p_MGN, p_wdb])
            )
        
        for subInd, subPath in enumerate(self.path_subjects[self.ind_start:], start=self.ind_start):
            print("processing %d-th subject, %.2f%% accomlished."%
                  (subInd, (subInd+1)*100/self.MGNSize_main))
            
            ## <===== compute per-vertex offsets for each sub.
            ## read registered dressed body
            subObj  = Mesh(filename = pjn(subPath, 'smpl_registered.obj'))  
            subTex  = pjn(subPath, 'registered_tex.jpg')  # texture map for body
            subSeg  = pjn(subPath, 'segmentation.png')    # segmentation map for body 
            
            subObj.set_texture_image(subSeg)  # prepare for seg_filter
            self.stdSMPLmesh.set_texture_image(subSeg)
            
            ## compute offsets directly 
            tarPara = [self.poses[subInd], self.betas[subInd], self.trans[subInd]]
            offsets_dir_t  = self.computeOffsets_direct(subObj, tarPara)       # compute t-offset of the whole body from GT obj
            offsets_dir_tf = seg_filter(subObj, offsets_dir_t)                 # remove meaningless offsets (esp. not covered)
            
            offsets_std_t = self.downMat.dot(offsets_dir_t.ravel()[:20670]).reshape(-1,3)   # downsample offsets to standard smpl mesh
            offsets_std_tf = seg_filter(self.stdSMPLmesh, offsets_std_t)
            
            # vis to debug, can be removed
            if self.verbose_on:
                print("show mesh in self.generateDataset(), for sub.")
                offsetList = [offsets_dir_t, offsets_dir_tf, offsets_std_tf]
                self.vis_offsets_debug(offsetList, tarPara, row=1, col=3)
            
            if self.save_offsets:
                savePath = pjn(subPath, 'gt_offsets/')
                save_offsets(offsets_dir_tf, offsets_std_tf, savePath)
            
            ## <===== create augmented/new subjects
            for poseInd in augPoseInd[subInd]:
                ## randomly choose suits for the body
                coatPathList, pantsPathList = self.setSuit(subInd, each_Nsuit, poseInd)
                
                for coatPath, pantsPath in zip(coatPathList, pantsPathList):
                    ## get offsets in t pose for both hres and std model
                    tarPara = [self.poses[poseInd], self.betas[poseInd], self.trans[poseInd]]
                    offset_t_hre = self.computeOffsets_guided(coatPath, pantsPath, tarPara, subObj, subTex)
                    
                    offset_t_std = self.computeOffsets_guided(
                        coatPath, pantsPath, tarPara, self.stdSMPLmesh, subTex, is_hres=False
                        )
                    
                    ## remove offsets according to segmentation
                    offset_t_hre_fil = seg_filter(subObj, offset_t_hre)

                    offset_t_std_fil = seg_filter(self.stdSMPLmesh, offset_t_std)
                    
                    # vis to debug, can be removed
                    if self.verbose_on:
                        print("show mesh in self.generateDataset(), for augmented.")
                        offsetList = [offset_t_hre, offset_t_std, offset_t_hre_fil, offset_t_std_fil]
                        self.vis_offsets_debug(offsetList, tarPara, row=2, col=2)
                    
                    if self.save_offsets:
                        posePath = self.path_subjects[poseInd] \
                            if poseInd < self.MGNSize_main \
                                else self.path_wardrobe[poseInd-self.MGNSize_main]
                            
                        # prepare the registered body mesh
                        smpl = smplFromParas(self.hresSMPL, offset_t_hre_fil, 
                                             tarPara[0], tarPara[1], tarPara[2])
                        subBody_hres = Mesh(smpl.r, smpl.f)
                        subBody_hres.vt = subObj.vt
                        subBody_hres.ft = subObj.ft
                        
                        # create and save the augmented subjects
                        savePath = create_subject(subPath, coatPath, pantsPath, posePath, subBody_hres)
                        savePath = pjn(savePath, 'gt_offsets/')
                        save_offsets(offset_t_hre_fil, offset_t_std_fil, savePath)
                    
    def vis_offsets_debug(self, offsetsList, posePara, row=1, col=2):
        """visualize the offsets in the give pose; for debug."""
        
        assert len(offsetsList) == row*col, 'please set row and col to display meshes.'
        
        bodyList = []
        
        for offset in offsetsList:
            if offset.shape[0] > 6890:
                smpl = smplFromParas(self.hresSMPL, offset, posePara[0], posePara[1], posePara[2])
                hresbody = Mesh(smpl.r, smpl.f)
                bodyList.append(hresbody)
            else:
                smpl = smplFromParas(self.stdSMPL,  offset, posePara[0], posePara[1], posePara[2])
                stdbody = Mesh(smpl.r, smpl.f)
                bodyList.append(stdbody)
        
        mvs = MeshViewers((row, col))
        for r in range(row):
            for c in range(col):
                mvs[r][c].set_static_meshes([ bodyList[r*col + c] ])
        
        
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_cfg', action='store', type=str, 
                        help='Path to the configuration for rendering.', 
                        default = './dataset_cfg.yaml')
    args = parser.parse_args()

    # read preparation configurations
    with open(args.dataset_cfg) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
        print('\nDataset preparation configurations:\n')
        for key, val in cfgs.items():
            if 'enable' in key:
                print('\n')
            print('%-25s:'%(key), val)    
    
    # require confirmation
    if cfgs['requireConfirm']:
        msg = 'Do you confirm that the settings are correct?'
        assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'    
    
    # create enriched MGN dataset
    if cfgs['enable_offsets']:
        mgn  = MGN_bodyAug_preparation(cfgs)
        mgn.generateDataset()
        
    print("Done")