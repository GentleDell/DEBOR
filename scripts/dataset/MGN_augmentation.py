#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:19:32 2021

@author: Zhantao
"""

import argparse
from os import makedirs
from os.path import abspath, isfile, exists, join as pjn
import sys 
if abspath('../') not in sys.path:
    sys.path.append(abspath('../'))
if abspath('./MGN_helper') not in sys.path:
    sys.path.append(abspath('./MGN_helper'))
if abspath('./MGN_helper/lib') not in sys.path:
    sys.path.append(abspath('./MGN_helper/lib'))
if abspath('./MGN_helper/utils') not in sys.path:
    sys.path.append(abspath('./MGN_helper/utils'))
import errno
from shutil import copyfile
from math import floor
import pickle as pkl
from glob import glob

import yaml
import torch
from torch import Tensor
import numpy as np
from psbody.mesh import Mesh, MeshViewers

from MGN_helper.utils.smpl_paths import SmplPaths
from ch_smpl import Smpl
from interpenetration_ind import remove_interpenetration_fast

_neglectParts_ = {'face and hand': (0, 0, 255),
                  'foots': (255, 0, 0),
                  'hairs': (0, 255, 0)
                  }

def check_folder(path: str):
    if not exists(path):
        try:
            makedirs(path)
        except OSError as exc:    # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise         

def compute_offset_tPose(smpl, garPath, thresholds, num_separations, enableVis = False):
    
    ## Get the original body mesh of the garment
    garFolder = '/'.join(garPath.split('/')[:-1])
    orig_body = pkl.load(open(pjn(garFolder, 'registration.pkl'), 'rb'), encoding='iso-8859-1')
    
    smpl = smplFromParas(smpl, np.zeros_like(smpl.r), 0, orig_body['betas'], 0)
    garment_org_body_unposed = Mesh(smpl.r, smpl.f)

    ## Get the original garment
    garment_unposed = Mesh(filename = garPath)
    
    ## remove interpenetration
    garment_unposed_interp = remove_interpenetration_fast(garment_unposed, garment_org_body_unposed)
    
    ## compute normal guided displacement
    body_normal = Tensor(garment_org_body_unposed.estimate_vertex_normals())
    
    offsets_tPose = computeNormalGuidedDiff(
        Tensor(garment_org_body_unposed.v), 
        body_normal, 
        Tensor(garment_unposed_interp.v), 
        Tensor(garment_unposed_interp.f.copy().astype('int32')).long(),
        num_separation = num_separations, 
        max_displacement = thresholds['max_offsets']
        )
    
    offsets_tPose = meshDisplacementFilter(
        Tensor(offsets_tPose), 
        Tensor(garment_org_body_unposed.f.copy().astype('int32')).long(),
        filterThres = thresholds['offsets_check_threshold'],
        diffThreshold = thresholds['max_diffOffsets']
        )
    
    if enableVis:
        print('show mesh in compute_offset_tPose().')
        v = garment_org_body_unposed.v + offsets_tPose
        body = Mesh(v, garment_org_body_unposed.f)
        
        mvs = MeshViewers((1, 1))
        mvs[0][0].set_static_meshes([body])
            
    return offsets_tPose

def computeNormalGuidedDiff(srcVerts: Tensor, srcNormal: Tensor, 
                            dstVerts: Tensor, triangle: Tensor,
                            num_separation: int, max_displacement: float) -> Tensor: 
    '''
    This function computes the displacements of vertices along their normal.

    Parameters
    ----------
    srcVerts : Tensor
        The SMPL vertices, [Nx3].
    dstVerts : Tensor
        The mesh of the registered scan, [Nx3].

    Returns
    -------
    Tensor
        The displacements of the given pointcloud to the given mesh.

    '''
    assert srcVerts.shape[1] == 3 and dstVerts.shape[1] == 3, \
        "src Vertices shape should be Nx3; dstVerts should be Nx3."
    assert (srcVerts.shape[0]/num_separation).is_integer(), \
        "vertices must divide batches to get an integer."
    
    dstVerts = Tensor(dstVerts[triangle.flatten()].reshape([-1, 9]))
    
    srcSize = srcVerts.shape[0]
    dstSize = dstVerts.shape[0]
    
    # use the Moller Trumbore Algorithm to find the displacements
    # to reduce memory footprint, we separate the point cloud into batches.
    P0, P1, P2 = dstVerts[:,:3], dstVerts[:,3:6], dstVerts[:,6:]
    E1 = P1 - P0
    E2 = P2 - P0

    fullDisps = []
    segLength = int(srcSize/num_separation)
    for segCnt in range(num_separation):
        
        # get the subset of vertices and normal vectors
        tempVertices = srcVerts[ segCnt*segLength : (segCnt+1)*segLength, : ]
        tempNormalVec= srcNormal[ segCnt*segLength : (segCnt+1)*segLength, : ]
                       
        # intermediate variables
        S  = tempVertices[:,None,:] - P0[None,:,:]
        S1 = tempNormalVec[:,None,:].repeat([1,dstSize,1]).cross(E2[None,:,:].repeat([segLength,1,1]))
        S2 = S.cross(E1[None,:,:].repeat([segLength,1,1]))
        
        # auxilary variables
        reciVec = 1/(S1[:,:,None,:].matmul(E1[None,:,:,None])).squeeze()
        t_disps = S2[:,:,None,:].matmul(E2[None,:,:,None]).squeeze()
        b1_disp = S1[:,:,None,:].matmul(S[:,:,:,None]).squeeze()
        b2_disp = S2[:,:,None,:].matmul(tempNormalVec[:,None,:,None]).squeeze()
        dispVec = reciVec[None,:,:] * torch.stack( (t_disps, b1_disp, b2_disp), dim = 0)
        
        # t: distance to the intersection points
        # b1 and b2: weight of E1 and E2
        t, b1, b2 = dispVec[0], dispVec[1], dispVec[2]
    
        # filter invalid intersection points outside the triplets and those
        # are too far away from the source points.
        # Here we allow small negative dist because:
        #     if we do not allow neg dist, when there were interpenetrations    {interpenetration free is not guranteed by the funciton}
        #        between garment and body in inner side of legs, leg verts      {remove_interpenetration_fast() of MGN implementation. }
        #        can only find facets from the other leg and create artifacts.
        #     if we allow large neg dist, verts can find facets on the other    {in the wardrobe, some pants and shirts only have half }
        #        side, which corr. to verts on the other side. This can result  {side mesh, not sure why but we need to work aroud this}
        #        in void body parts, e.g. missing ankle.
        # So, allwoing a proper small neg dist could be a good solution and we
        # choose 1cm.
        intersection_mask = (b1 > 0).logical_and(b2 > 0)      \
                                    .logical_and(b1+b2 < 1)   \
                                    .logical_and(t > -0.01)   \
                                    .logical_and(t.abs() < max_displacement)    
        
        # choose the closest displacement if not unique
        indice = torch.nonzero(intersection_mask,as_tuple=False)
        subindice, cnt = torch.unique(indice[:,0], return_counts=True)
        for ambInd in torch.nonzero(cnt > 1, as_tuple=False):
            
            src_dst_repeat = torch.nonzero(indice[:, 0] == subindice[ambInd], as_tuple=False)
            keepsubInd = t[subindice[ambInd], indice[src_dst_repeat,1]].abs().argmin()
            
            keepInd = src_dst_repeat[keepsubInd]
            dropInd = torch.cat( (src_dst_repeat[:keepsubInd], src_dst_repeat[keepsubInd+1:]) )

            indice[dropInd,:] = indice[keepInd, :] 
            
        # convert displacements to vectors
        partialDisp = torch.zeros(segLength, 3)
        if indice.shape[0] != 0:
            indice = indice.unique(dim = 0)
            distance = t[indice[:,0], indice[:,1]] 
            normals  = tempNormalVec[indice[:,0]]
            displace = distance[:,None]*normals
            
            # fill data
            partialDisp[indice[:,0],:] = displace
            
        fullDisps.append( partialDisp )
    
    return torch.cat(fullDisps, dim = 0).numpy()

def meshDisplacementFilter(displacements: Tensor, triplets: Tensor,
                           filterThres: float, diffThreshold: float):
    '''
    This function filters offsets suspected to be outliers in the given 
    displacements according to the given filterThres.
    
    The filterThres specifies the suspects. Then the displacements of the NN 
    of suspectes are collected. If the difference between the displacements of 
    suspects and the median of their NN is greater than the diffThreshold, the 
    displacements of the suspects will be replaced by the median.

    Ideally, it would be better to filter all points instead of focusing on 
    several obvious outliers, but full filtering would take time/memory, so 
    currently, we suggest not to do full filtering.
    
    Parameters
    ----------
    displacements : Tensor
        The displacements to be filtered, [Nx3].
    triplets : Tensor
        The mesh of the body, [Nx3].

    Returns
    -------
    displacements : TYPE
        The filtered displacements.

    '''     
    o_offset = displacements.clone()
    suspects = torch.nonzero(displacements.norm(dim=1) > filterThres, as_tuple=False)
    for pointInd in suspects:
        NNtriplets = torch.nonzero((triplets == pointInd).sum(dim=1), as_tuple=False)
        NNpointsId = triplets[NNtriplets].unique().long()
        vals, inds = displacements[NNpointsId].norm(dim=1).sort()

        medianInd  = floor(vals.shape[0]/2) - 1
        if medianInd > 0:
            if (displacements[pointInd].norm() - vals[medianInd] ).abs() > diffThreshold:
                o_offset[pointInd] = displacements[ NNpointsId[inds[medianInd]] ]

    return o_offset.numpy()

def seg_filter(subObj, offsets):
  
    ## get per-vertex color as segmentation
    verts_allUV = subObj.texture_coordinates_by_vertex()
    verts_uvs = np.array([uv[0] for uv in verts_allUV])
    verts_seg = subObj.texture_rgb_vec(verts_uvs)
    
    ## remove offsets of foots, hand arm and face
    outOffset = offsets.copy()
    for part, color in _neglectParts_.items():
        mask = (verts_seg[:,0] == color[0])* \
               (verts_seg[:,1] == color[1])* \
               (verts_seg[:,2] == color[2])
        outOffset[mask] = 0
    
    return outOffset

def create_subject(subPath, coatPath, pantsPath, posePath, subBody_hres):
    
    ## prepare path
    coatID  = coatPath.split('/')[-2][-3:]
    pantsID = pantsPath.split('/')[-2][-3:]
    if 'Multi-Garment_dataset' in posePath:
        pose_ID = posePath.split('/')[-1][-4:]
    elif 'MGN_dataset_02' in posePath:
        pose_ID = posePath.split('/')[-2][-3:]
    else:
        raise ValueError('please make sure the name of folder is correct')
    path = '_'.join([subPath, 'coat', coatID, 'pants', pantsID, 'pose', pose_ID])
    
    check_folder(path)

    ## move reated files to the folder     
    copyfile(pjn(subPath, 'multi_tex.jpg'), pjn(path, 'multi_tex.jpg'))
    copyfile(pjn(subPath, 'registered_tex.jpg'), pjn(path, 'registered_tex.jpg'))
    copyfile(pjn(subPath, 'segmentation.png'), pjn(path, 'segmentation.png'))
    
    coat_name = coatPath.split('/')[-1]
    copyfile(coatPath, pjn(path, coat_name))
    pants_name= pantsPath.split('/')[-1]
    copyfile(pantsPath, pjn(path, pants_name))
    
    copyfile(pjn(posePath,'registration.pkl'), pjn(path,'registration.pkl'))
          
    ## save the registered mesh
    if isinstance(subBody_hres, Mesh):
        subBody_hres.write_obj( pjn(path, 'smpl_registered.obj') )
    else:
        raise ValueError('subBody_hres should be a Mesh object.')
    
    return path

def save_offsets(offsets_hres, offsets_std, savePath):
    
    check_folder(savePath)
    
    with open(pjn(savePath,'offsets_hres.npy'), 'wb') as f:
        np.save(f, offsets_hres)             
    with open(pjn(savePath, 'offsets_std.npy'), 'wb') as f:
        np.save(f, offsets_std)       

def smplFromParas(smpl, offsets, pose, betas, trans):
    
    smpl.v_personal[:] = offsets
    smpl.pose[:]  = pose
    smpl.betas[:] = betas
    smpl.trans[:] = trans
    
    return smpl

class MGN_bodyAug_preparation(object):
        
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
            'Pants': [],
            'ShortPants': [],
            'ShirtNoCoat': [],
            'TShirtNoCoat': [], 
            'LongCoat': []
            }
        # inproper garments:
        #   1. tightly-coupled with pose
        #   2. having strange/incorrect structure, e.g. incorrect segmentation 
        #   3. involve irrelevant parts as garments, e.g. hair
        skipSamples = ['005', '006', '017', '023', '025', '028', '030', '038', '041',
                       '051', '054', '059', '085', '099']
        for garments in self.path_wardrobe:
            # collect smpl parameters in wardrobe to enrich the dataset
            pathRegistr  = pjn(garments, 'registration.pkl')
            registration = pkl.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( torch.Tensor(registration['betas'][None,:]) )
            self.poses.append( torch.Tensor(registration['pose'][None,:]) )
            self.trans.append( torch.Tensor(registration['trans'][None,:]) )
            
            if garments[-3:] in skipSamples:
                continue    # skip inproper garments samples
            
            suit = '  '.join(glob(pjn(garments, '*.obj')))
            if 'ShortPants' in suit:
                self.wardrobe['ShortPants'].append(garments)
            elif 'Pants' in suit:
                self.wardrobe['Pants'].append(garments)
            if 'TShirtNoCoat' in suit:
                self.wardrobe['TShirtNoCoat'].append(garments)
            elif 'ShirtNoCoat' in suit:
                self.wardrobe['ShirtNoCoat'].append(garments)
            elif 'LongCoat' in suit:
                self.wardrobe['LongCoat'].append(garments)
        
        ## <==== prepare high resolution SMPL, standard SMPL and downsampling mat
        dp = SmplPaths()
        self.hresSMPL = Smpl(dp.get_hres_smpl_model_data())
        self.stdSMPL  = Smpl(dp.get_smpl_file())
        self.downMat  = np.load(cfg['path_downMat'])
        self.stdSMPLmesh = Mesh(filename=cfg['smplMesh_std'])    # std res mesh of SMPL model
        
        ## <==== get configurations for post processing
        self.verbose_on = cfg['show_intermediateMeshes']
        self.save_offsets = cfg['save_displacements']
        self.num_separation = {'hres': cfg['number_separation_hres'],
                               'std' : cfg['number_separation_std' ]}
        self.thresholds = {'coats': {}, 'pants':{}}
        for ind, key in enumerate(sorted(self.thresholds.keys())):
            self.thresholds[key]['max_offsets'] = cfg['max_offsets'][ind]
            self.thresholds[key]['offsets_check_threshold'] = cfg['offsets_check_threshold'][ind]
            self.thresholds[key]['max_diffOffsets'] = cfg['max_diffOffsets'][ind]
    
    def setSuit(self, subInd: int, each_Nsuit: int):
        # We keep the augmented garments to have the same style as the 
        # original garment as we need to use its texture.
        
        # get the original garment style
        origSuit = self.subjectSuit[subInd]
        
        # randomly choose pants
        pantsInd = np.random.randint(len(self.wardrobe[origSuit[0]]), size = each_Nsuit).tolist()
        pantsPath= [pjn(self.wardrobe[origSuit[0]][ind], origSuit[0]+'.obj') for ind in pantsInd]
        
        # randomly choose coats for the body 
        if 'Coat' not in '*'.join(origSuit):
            # if a subject does not have coats, from what we observe, it 
            # probably wears a tshir, except for the 83rd which is naked.
            if subInd == 83:
                coatPath = None    # naked
            else:
                coatInd  = np.random.randint(len(self.wardrobe['TShirtNoCoat']), size = each_Nsuit).tolist()
                coatPath = [pjn(self.wardrobe['TShirtNoCoat'][ind], 'TShirtNoCoat.obj') for ind in coatInd]
        else:
            # pick a coat in the same style of the orignal coat
            coatInd  = np.random.randint(len(self.wardrobe[origSuit[1]]), size = each_Nsuit).tolist()
            coatPath = [pjn(self.wardrobe[origSuit[1]][ind], origSuit[1]+'.obj') for ind in coatInd]
        
        return coatPath, pantsPath
    
    def computeOffsets_guided(
            self, coatPath: str, pantsPath: str, tarPara: list, 
            subObj: Mesh = None, subTex: str = None, is_hres: bool = True):
        
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
        
        ## choose additional poses for all subjects in MGN
        augPoseInd = np.random.choice(np.arange(len(self.poses)), size=(len(self.path_subjects), each_Npose))
        
        for subInd, subPath in enumerate(self.path_subjects):
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
            
            ## <===== create augmented subjects
            for poseInd in augPoseInd[subInd]:
                ## randomly choose a suit for the body
                coatPathList, pantsPathList = self.setSuit(subInd, each_Nsuit)
                
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
                            if poseInd < len(self.path_subjects) \
                                else self.path_subjects[poseInd-len(self.path_subjects)]
                            
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

    mgn  = MGN_bodyAug_preparation(cfgs)
    mgn.generateDataset()