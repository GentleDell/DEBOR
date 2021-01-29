#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:20:47 2021

@author: zhantao
"""

import pickle as pkl
from shutil import copyfile
from math import floor
import errno
from os import makedirs
from os.path import exists, join as pjn

import torch
from torch import Tensor
import numpy as np
from psbody.mesh import Mesh, MeshViewers

from interpenetration_ind import remove_interpenetration_fast

_neglectParts_ = {'face and hand': (0, 0, 255),
                  'foots': (255, 0, 0),
                  'hairs': (0, 255, 0)
                 }

def check_folder(path: str):
    """check if the path exist. If not, ceate one."""
    if not exists(path):
        try:
            makedirs(path)
        except OSError as exc:    # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise   

def smplFromParas(smpl, offsets, pose, betas, trans):
    """set the smpl model with the given parameters."""
    smpl.v_personal[:] = offsets
    smpl.pose[:]  = pose
    smpl.betas[:] = betas
    smpl.trans[:] = trans
    
    return smpl

def compute_offset_tPose(smpl, garPath, thresholds, num_separations, enableVis = False):
    """compute the per-vertex offsets in t-pose using the normal guided method."""
    
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
    Same as the function in mesh_util.meshDifferentiator().

    Parameters
    ----------
    srcVerts : Tensor
        The SMPL vertices, [Nx3].
    srcNormal: Tnesor 
        The normal of the given srcVerts.
    dstVerts : Tensor
        The mesh of the registered scan, [Nx3].
    triangle : Tenxor
        The indices of vertice of dstVerts to define its mesh.
    num_separation: int
        To reduce memory usage, we separate the mesh to num_separation parts.
    max_displacement: float
        Threshold to control the computed offsets.

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

    Same as the function in mesh_util.meshDifferentiator().
    
    Parameters
    ----------
    displacements : Tensor
        The displacements to be filtered, [Nx3].
    triplets : Tensor
        The mesh of the body, [Nx3].
    filterThres: float
        The threshold to decide if an offset is suspicious.
    diffThreshold: float
        The threshold to decide if an offset should be replaced. 

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
    """use segmentation map to filter offsets."""
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
    """manage the files of the augmented/new subject."""
    
    ## prepare path
    coatID  = coatPath.split('/')[-2][-3:] if coatPath is not None else 'none'
    pantsID = pantsPath.split('/')[-2][-3:]
    if 'Multi-Garment_dataset' in posePath:
        pose_ID = posePath.split('/')[-1][-4:]
    elif 'MGN_dataset_02' in posePath:
        pose_ID = posePath.split('/')[-1][-3:]
    else:
        raise ValueError('please make sure the name of folder is correct')
    path = '_'.join([subPath, 'coat', coatID, 'pants', pantsID, 'pose', pose_ID])
    
    check_folder(path)

    ## move reated files to the folder     
    copyfile(pjn(subPath, 'multi_tex.jpg'), pjn(path, 'multi_tex.jpg'))
    copyfile(pjn(subPath, 'registered_tex.jpg'), pjn(path, 'registered_tex.jpg'))
    copyfile(pjn(subPath, 'segmentation.png'), pjn(path, 'segmentation.png'))
    
    if coatPath is not None:    # for the 83rd subject
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