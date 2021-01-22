#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:19:32 2021

@author: Zhantao
"""

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
from math import floor
import pickle as pkl
from glob import glob

import torch
from torch import Tensor
import numpy as np
from psbody.mesh import Mesh, MeshViewers

from MGN_helper.utils.smpl_paths import SmplPaths
from ch_smpl import Smpl
from interpenetration_ind import remove_interpenetration_fast

_neglectParts_ = {'face and hand': (0,0,255),
                  'foots': (255, 0, 0) 
                  }

def load_smpl_from_file(file):
    dat = pkl.load(open(file, 'rb'), encoding='iso-8859-1')
    dp = SmplPaths(gender=dat['gender'])
    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    smpl_h.pose[:] = dat['pose']
    smpl_h.betas[:] = dat['betas']
    smpl_h.trans[:] = dat['trans']

    return smpl_h

def compute_offset_tPose(garPath, vert_inds, smpl):
    
    ## Get the original body mesh of the garment
    garFolder = '/'.join(garPath.split('/')[:-1])
    garment_org_body_unposed = load_smpl_from_file(pjn(garFolder, 'registration.pkl'))
    garment_org_body_unposed.pose[:] = 0
    garment_org_body_unposed.trans[:] = 0
    garment_org_body_unposed = Mesh(garment_org_body_unposed.v, garment_org_body_unposed.f)
    
    ## Get the original garment
    garment_unposed = Mesh(filename = garPath)
    
    ## remove interpenetration
    # garment_unposed_interp = remove_interpenetration_fast(garment_unposed, garment_org_body_unposed)

    ## compute normal guided displacement
    body_normal = Tensor(garment_org_body_unposed.estimate_vertex_normals())
    offsets_tPose = computeNormalGuidedDiff(
        Tensor(garment_org_body_unposed.v), 
        body_normal, 
        Tensor(garment_unposed.v), 
        Tensor(garment_unposed.f.copy().astype('int32')).long(),
        num_separation = 23, 
        max_displacement = 0.2
        )
    
    offsets_tPose = meshDisplacementFilter(
        Tensor(offsets_tPose), 
        Tensor(garment_unposed.f.copy().astype('int32')).long(),
        filterThres = 0.05,
        diffThreshold = 0.03
        )
    
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
        # are far away from the source points(SMPL vertices)
        intersection_mask = (b1 > 0).logical_and(b2 > 0)      \
                                    .logical_and(b1+b2 < 1)   \
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

def seg_filter(subObj, segmentation, offsets):
    ## load segmentation
    subObj.set_texture_image(segmentation)

    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([subObj])
    
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

def save_offset(offsets_hres, offsets_std, savePath):
    if not exists(savePath):
        try:
            makedirs(savePath)
        except OSError as exc:    # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise         
    with open('offsets_hres.npy', 'wb') as f:
        np.save(f, offsets_hres)             
    with open('offsets_std.npy', 'wb') as f:
        np.save(f, offsets_std)       

class MGN_bodyAug_preparation(object):
        
    def __init__(self, 
                 path_MGN_major: str, 
                 path_MGN_wardrobe: str, 
                 pathSMPL: str,
                 path_downMat: str,
                 fts_file: str):
                
        self.path_smpl = pathSMPL
        self.path_subjects = sorted( glob(pjn(path_MGN_major, '*')) )
        self.path_wardrobe = sorted( glob(pjn(path_MGN_wardrobe, '*')) )
        
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
        # challenging coats: 035, 054
        self.wardrobe = {
            'Pants': [],
            'ShortPants': [],
            'ShirtNoCoat': [],
            'TShirtNoCoat': [], 
            'LongCoat': []
            }
        skipSamples = ['023', '025', '028', '030', '038', '041', '055', '059', '085', '099']
        for garments in self.path_wardrobe:
            # collect smpl parameters in wardrobe to enrich the dataset
            pathRegistr  = pjn(garments, 'registration.pkl')
            registration = pkl.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
            self.betas.append( torch.Tensor(registration['betas'][None,:]) )
            self.poses.append( torch.Tensor(registration['pose'][None,:]) )
            self.trans.append( torch.Tensor(registration['trans'][None,:]) )
            
            if garments[-3:] in skipSamples:
                continue    # skip inproper samples
            
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
        
        ## prepare target body
        dp = SmplPaths()
        vt, ft = dp.get_vt_ft_hres()
        self.smpl = Smpl(dp.get_hres_smpl_model_data())
        
        ## this file contains correspondances between garment vertices and smpl body
        self.vert_indices, self.fts = pkl.load(open(fts_file, 'rb'), encoding='iso-8859-1')
        self.fts['naked'] = ft
        
        ## this file contains matrix to convert hres mesh to std smpl mesh (t-pose/0-pose only) 
        ## get standard mesh for verification, can be removed
        self.downMat = np.load(path_downMat)
        self.stdSMPL = Smpl(dp.get_smpl_file())
    
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
    
    def dressBody(self, coatPath: str, pantsPath: str, tarPara: list, 
                  subObj: Mesh, garmtex: str = None, subTex: str = None):
        
        ## Generate target SMPL body 
        self.smpl.pose[:]  = tarPara[0]
        self.smpl.betas[:] = tarPara[1]
        self.smpl.trans[:] = 0    # trans of the garments are set to 0, so we need to set thi to 0.
        self.smpl.v_personal[:] = np.zeros_like(self.smpl.r)
        
        ## per-vertex offsets
        v_offsets_t = np.zeros_like(self.smpl.r)
        
        ## Pants
        pantsStyle = pantsPath.split('/')[-1][:-4]
        v_pants_inds = self.vert_indices[pantsStyle]
        offset_pants_t = compute_offset_tPose(pantsPath, v_pants_inds, self.smpl)
        
        mask = np.linalg.norm(offset_pants_t, axis=1) > np.linalg.norm(v_offsets_t, axis=1)    # keep the greater distance
        v_offsets_t[mask] = offset_pants_t[mask]
        
        ## coat
        # None for sub84 and others subs without coat in their folder
        if coatPath is not None:    
            coatStyle = coatPath.split('/')[-1][:-4]
            v_coat_inds = self.vert_indices[coatStyle]
            offset_coat_t = compute_offset_tPose(coatPath, v_coat_inds, self.smpl)
            
            mask = np.linalg.norm(offset_coat_t, axis=1) > np.linalg.norm(v_offsets_t, axis=1)    # keep the greater distance
            v_offsets_t[mask] = offset_coat_t[mask]
        
        ## Dress body
        self.smpl.pose[:]  = tarPara[0]
        self.smpl.betas[:] = tarPara[1]
        self.smpl.trans[:] = tarPara[2]
        self.smpl.v_personal[:] = v_offsets_t
        dressed_body = Mesh(self.smpl.r, self.smpl.f)
        dressed_body.vt = subObj.vt
        dressed_body.ft = subObj.ft
        dressed_body.set_texture_image(subTex)
        
        mvs = MeshViewers((1, 1))
        mvs[0][0].set_static_meshes([dressed_body])
        
        return v_offsets_t, dressed_body
    
    def computeOffset_direct(self, subObj, tarPara):
        ## only works when the registered subject is available
        
        ## compute the posed offsets
        self.smpl.v_personal[:] = np.zeros_like(self.smpl.r)
        self.smpl.pose[:]  = tarPara[0]
        self.smpl.betas[:] = tarPara[1]
        self.smpl.trans[:] = tarPara[2]
        offsets_p = subObj.v - self.smpl.r
        
        ## unpose the offsets
        # 1. put on the offsets to body and do inverse pose
        self.smpl.v_personal[:] = offsets_p
        self.smpl.pose[:]  = -tarPara[0]    # inverse is simply minors
        self.smpl.betas[:] = tarPara[1]
        self.smpl.trans[:] = tarPara[2]
        invposed_body_off = self.smpl.r
        
        # 2. generate naked body in inverse pose
        self.smpl.v_personal[:] = np.zeros_like(self.smpl.r)
        self.smpl.pose[:]  = -tarPara[0]
        self.smpl.betas[:] = tarPara[1]
        self.smpl.trans[:] = tarPara[2]
        invposed_body = self.smpl.r
        
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
            garmtex = pjn(subPath, 'multi_tex.jpg')       # texture map for garments
            
            ## compute offsets directly 
            tarPara = [self.poses[subInd], self.betas[subInd], self.trans[subInd]]
            offsets_dir_t  = self.computeOffset_direct(subObj, tarPara)       # compute t-offset of the whole body from GT obj
            offsets_dir_tf = seg_filter(subObj, subSeg, offsets_dir_t)      # remove meaningless offsets (esp. not covered)
            offsets_std_tf = self.downMat.dot(offsets_dir_tf.ravel()[:20670]).reshape(-1,3)   # downsample offsets to standard smpl mesh
            
            # vis to debug, can be removed
            offsetList = [offsets_dir_t, offsets_dir_tf, offsets_std_tf]
            self.vis_offsets_debug(offsetList, tarPara, row=1, col=3)
                        
            ## <===== create augmented subjects
            for poseInd in augPoseInd[subInd]:
                ## randomly choose a suit for the body
                coatPathList, pantsPathList = self.setSuit(subInd, each_Nsuit)
                
                considering different threshold for coat and pants, even for diff coats
                
                for coatPath, pantsPath in zip(coatPathList, pantsPathList): 
                    ## get the offset in t pose
                    tarPara = [self.poses[poseInd], self.betas[subInd], self.trans[subInd]]
                    offset_t, dressedSub = self.dressBody(coatPath, pantsPath, tarPara, subObj, garmtex, subTex)
                    
                    offset_t_std = self.downMat.dot(offset_t.ravel()[:20670]).reshape(-1,3)
                    
                    ## remove offsets according to segmentation
                    offset_t_fil = seg_filter(subObj, subSeg, offset_t)
                    ## downsample the offsets to standard smpl mesh resolution
                    offset_t_std_fil = self.downMat.dot(offset_t_fil.ravel()[:20670]).reshape(-1,3)
                    
                    # vis to debug, can be removed
                    offsetList = [offset_t, offset_t_std, offset_t_fil, offset_t_std_fil]
                    self.vis_offsets_debug(offsetList, tarPara, row=2, col=2)
                    
                    
    def vis_offsets_debug(self, offsetsList, posePara, row=1, col=2):
        
        assert len(offsetsList) == row*col, 'please set row and col to display meshes.'
        
        bodyList = []
        
        for offset in offsetsList:
            if offset.shape[0] > 6890:
                self.smpl.v_personal[:] = offset
                self.smpl.pose[:]  = posePara[0]
                self.smpl.betas[:] = posePara[1]
                self.smpl.trans[:] = posePara[2]
                hresbody = Mesh(self.smpl.r, self.smpl.f)
                bodyList.append(hresbody)
            else:
                self.stdSMPL.v_personal[:] = offset
                self.stdSMPL.pose[:]  = posePara[0]
                self.stdSMPL.betas[:] = posePara[1]
                self.stdSMPL.trans[:] = posePara[2]
                stdbody = Mesh(self.stdSMPL.r, self.stdSMPL.f)
                bodyList.append(stdbody)
        
        mvs = MeshViewers((row, col))
        for r in range(row):
            for c in range(col):
                mvs[r][c].set_static_meshes([ bodyList[r*col + c] ])
                    
        
if __name__ == "__main__":

    pathmodel = '/home/logix/Documents/DEBOR/body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    path = '../../datasets/Multi-Garment_dataset'
    path_wardrobe = '../../datasets/MGN_dataset_02'
    path_fts_file = '../../body_model/garment_fts.pkl'
    path_downMat  = '../../body_model/downConvMat_MGN.npy'
    mgn  = MGN_bodyAug_preparation(path, path_wardrobe, pathmodel, path_downMat, path_fts_file)
    mgn.generateDataset()