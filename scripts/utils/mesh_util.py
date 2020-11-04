#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:11:39 2020

@author: zhantao
"""

import sys
sys.path.append('../')
sys.path.append('../third_party/smpl_webuser')
from os import makedirs
from os.path import join as pjn, isfile, exists
import pickle
import errno
from math import floor

from PIL import Image
import torch
from torch import Tensor

import open3d as o3d
import numpy as np
import chumpy as ch

if __name__ == "__main__":    # avoid relative path issue
    from vis_util import visDisplacement, read_Obj, interpolateTexture, visualizePointCloud
else:
    from .vis_util import visDisplacement, read_Obj, interpolateTexture, visualizePointCloud
from third_party.smpl_webuser.serialization import load_model
from third_party.smpl_webuser.lbs import verts_core as _verts_core 


_neglectParts_ = {'face and hand': (0,0,255),
                  'foots': (255, 0, 0) }


def meshDisplacementFilter(displacements: Tensor, triplets: Tensor, 
                           threshold: float = 0.06, 
                           diffThreshold: float = 0.02):
    
    suspects = torch.nonzero(displacements.norm(dim=1) > threshold)
    for pointInd in suspects:
        NNtriplets = torch.nonzero((triplets == pointInd).sum(dim=1))
        NNpointsId = triplets[NNtriplets].unique().long()
        vals, inds = displacements[NNpointsId].norm(dim=1).sort()
        
        medianInd  = floor(vals.shape[0]/2) - 1
        if (displacements[pointInd].norm() - vals[medianInd] ).abs() > diffThreshold:
            displacements[pointInd] = displacements[ NNpointsId[inds[medianInd]] ]
            
    return displacements


def computeNormalGuidedDiff(srcVertices: Tensor, dstMesh: Tensor, 
                            normalKNN: int=10, batches: int=10, 
                            maxDisp: float=0.1) -> Tensor: 
    '''
    This function computes the displacements of vertices along their normal
    vectors estimated by the open3d package.

    Parameters
    ----------
    srcVertices : Tensor
        The SMPL vertices, [6890x3].
    dstMesh : Tensor
        The mesh of the registered scan, [Nx9].
    normalKNN : int, optional
        The number of nearest neigbor to estimate normal for vertices.
        The default is 10.
    batches : int, optional
        In order to make memory footprint feasible, we sepqrate the pointcloud 
        to the given batches. The default is 10.
    maxDisp : float, optional
        The maximum acceptable displacements of each vertices. The larger the
        more points would be involved in filtering so that makes this function 
        slower.
        The default is 0.1.

    Returns
    -------
    Tensor
        The displacements of the given pointcloud to the given mesh.

    '''
    assert srcVertices.shape[1] == 3 and dstMesh.shape[1] == 9, "src Vertices shape should be Nx3; dstMesh should be Nx9."
    assert (srcVertices.shape[0]/batches).is_integer(), "vertices must divide batches to get an integer."
    
    KNNPara = o3d.geometry.KDTreeSearchParamKNN(normalKNN)
    srcSize = srcVertices.shape[0]
    dstSize = dstMesh.shape[0]

    # use open3d to estimate normals
    srcPointCloud = o3d.geometry.PointCloud()
    srcPointCloud.points = o3d.utility.Vector3dVector(srcVertices)
    srcPointCloud.estimate_normals(search_param = KNNPara)
    srcNormal = Tensor(srcPointCloud.normals)
    
    # use the Moller Trumbore Algorithm to find the displacements
    # to reduce memory footprint, we separate the point cloud into batches.
    P0, P1, P2 = dstMesh[:,:3], dstMesh[:,3:6], dstMesh[:,6:]
    E1 = P1 - P0
    E2 = P2 - P0

    fullDisps = []
    segLength = int(srcSize/batches)
    for segCnt in range(batches):
        
        # get the subset of vertices and normal vectors
        tempVertices = srcVertices[ segCnt*segLength : (segCnt+1)*segLength, : ]
        tempNormalVec= srcNormal[ segCnt*segLength : (segCnt+1)*segLength, : ]
           
        # debug
        # dbg_pc = torch.cat( (dstMesh.reshape(-1,3), tempVertices), dim = 0 )
        # colors = torch.zeros(dbg_pc.shape)
        # colors[:dstMesh.shape[0]*3, 0] = 1
        # colors[dstMesh.shape[0]*3:, 1] = 1
        # visualizePointCloud(dbg_pc, colors)
        
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
        intersection_mask = (b1 > 0).logical_and(b2 > 0).logical_and(b1+b2 < 1).logical_and(t.abs() < maxDisp)
        
        # choose the closest displacement if not unique
        indice = torch.nonzero(intersection_mask)
        subindice, cnt = torch.unique(indice[:,0], return_counts=True)
        for ambInd in torch.nonzero(cnt > 1):
            
            src_dst_repeat = torch.nonzero(indice[:, 0] == subindice[ambInd])
            keepsubInd = t[subindice[ambInd], indice[src_dst_repeat,1]].abs().argmin()
            
            keepInd = src_dst_repeat[keepsubInd]
            dropInd = torch.cat( (src_dst_repeat[:keepsubInd], src_dst_repeat[keepsubInd+1:]) )

            indice[dropInd,:] = indice[keepInd, :] 
            
        # convert displacements to vectors
        indice = indice.unique(dim = 0)
        distance = t[indice[:,0], indice[:,1]] 
        normals  = tempNormalVec[indice[:,0]]
        displace = distance[:,None]*normals
        
        # fill data
        partialDisp = torch.zeros(segLength, 3)
        partialDisp[indice[:,0],:] = displace
        fullDisps.append( partialDisp )
    
    return torch.cat(fullDisps, dim = 0)


def meshDifference_pointTopoint(srcVertices: Tensor, dstVertices: Tensor) -> Tensor: 
    '''
    This function computes the point-to-point difference between two meshes, 
    in the manner of dstVetices - srcVertices.

    Parameters
    ----------
    srcVertices : Tensor
        The vertices of the source mesh which we would like to estimate the 
        displacement.
    dstVertices : Tensor
        The vertices of the destination mesh from which we could like to 
        estimate the displacement.

    Returns
    -------
    Tensor
        The point-to-point displacement of srcVertices to fit the dstVetices.

    '''
    assert srcVertices.shape[1] == 3 and dstVertices.shape[1] == 3, 'Input shape should be Nx3'
    
    distanceMatrix = (srcVertices[:,None,:] - dstVertices[None, :,:]).norm(dim = 2)
    srcNearestInds = distanceMatrix.argmin(dim=1)
    srcNearestVert = dstVertices[srcNearestInds, :]
    
    meshDifference = srcNearestVert - srcVertices  
    
    return meshDifference


def meshDifference_pointTofacet(srcVertices: Tensor, dstTriangulars: Tensor,
                                usePlane: bool = False) -> Tensor:
    '''
    This function computes the distance of between the given srcVertices and 
    the given triangluar facets. If usePlane, the point-to-plane distance is 
    used, otherwise the point-to-triangleCenter distance is used.

    Since the point-to-plane distance does not evaluate the displacement well
    (closed mesh would always make the distance very small), it is recommended
    to use the point-to-triangleCenter distance.

    Parameters
    ----------
    srcVertices : Tensor
        The vertices of the source mesh which we would like to estimate the 
        displacement.
    dstTriangulars : Tensor
        The triangles of the destination mesh from which we could like to 
        estimate the displacement.
    usePlane : bool, optional
        Whether use the point-to-plane distance. The default is False.

    Returns
    -------
    Tensor
        The displacement of srcVertices to fit the dstVetices.

    '''
    assert srcVertices.shape[1] == 3 and dstTriangulars.shape[1] == 9, \
          'Src input shape should be Nx3 and dst input shape should be Mx9'
          
    p1, p2, p3 = dstTriangulars[:,:3], dstTriangulars[:,3:6], dstTriangulars[:,6:]
    
    if usePlane:
        # compute triangular facet normal
        vec12, vec13 = p2-p1, p3-p1
        normalVec  = vec12.cross(vec13)
        normalVec  = normalVec/normalVec.norm(dim=1)[:,None]
        
        # compute the signed distance to mesh facets
        vecDiff = srcVertices[:,None,:] - p1[None,:,:]
        signDist= torch.matmul(normalVec[None,:,None,:], vecDiff[:,:,:,None]).squeeze()
        
        # find the closest facet to each point
        values, indices = signDist.abs().min(dim=1)
        
        meshDifference = normalVec[indices] * values[:,None]
    else: 
        # compute the center of facet
        centers = (p1 + p2 + p3)/3
        meshDifference = meshDifference_pointTopoint(srcVertices, centers)
    
    return meshDifference
    

def computeDisplacement(path_objectFolder: str, path_SMPLmodel: str, 
                        dispThreshold: float, numNearestNeighbor: int,
                        onlyCloth: bool = True, NNfiltering: bool = False,
                        filterThres: float = 0.05, save: bool = True, 
                        device: str = 'cuda'):
    '''
    This function read the given object and SMPL parameter to compute the
    displacement.

    Parameters
    ----------
    path_objectFolder : str
        Path to the folder containing MGN objects.
    path_SMPLmodel : str
        Path to the SMPL models  file.
    dispThreshold: float, 
        The maximum acceptable displacements of each vertices. The larger the
        more points would be involved in filtering so that makes this function 
        slower.
    numNearestNeighbor: int
        The number of nearest neigbor to estimate normal for vertices.
    onlyCloth: bool 
        Whether only computes the displacements of clothes, i.e. remove the 
        displacements of unclothed parts.
        The default is True.
    save : bool, optional
        Whether to save the displacements.
        The default is True.
    device : str, optional
        The device where the computation will take place.
        The default is 'cuda'.

    Returns
    -------
    None.

    '''
    assert isfile(path_SMPLmodel), 'SMPL model not found.'
    assert device in ('cpu', 'cuda'), 'device = %s is not supported.'%device
    ondevice = ('cpu', device) [torch.cuda.is_available()]

    # prepare path to files
    objfile = pjn(path_objectFolder, 'smpl_registered.obj')
    regfile = pjn(path_objectFolder, 'registration.pkl')    # smpl model params
    segfile = pjn(path_objectFolder, 'segmentation.png')    # clothes segmentation 
    smplObj = pjn('/'.join(path_SMPLmodel.split('/')[:-1]), 'text_uv_coor_smpl.obj')    # clothes segmentation 

    # load registered mesh
    dstVertices, Triangle, _, _ = read_Obj(objfile)
    dstTriangulars = dstVertices[Triangle.flatten()].reshape([-1, 9])

    # load segmentaiton and propagate to meshes
    segmentations  = np.array(Image.open(segfile))
    smplVert, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(smplObj)
    segmentations  = interpolateTexture(smplVert.shape, smplMesh, segmentations, smpl_text_uv_coord, smpl_text_uv_mesh)
    # visualizePointCloud(smplVert, segmentations)

    # load SMPL parameters and model, transform vertices and joints.
    registration = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
    SMPLmodel = load_model(path_SMPLmodel)
    SMPLmodel.pose[:]  = registration['pose']
    SMPLmodel.betas[:] = registration['betas']
    [SMPLvert, joints] = _verts_core(SMPLmodel.pose, SMPLmodel.v_posed, SMPLmodel.J,  \
                                  SMPLmodel.weights, SMPLmodel.kintree_table, want_Jtr=True, xp=ch)
    joints = joints + registration['trans']
    SMPLvert = SMPLvert + registration['trans']

    # compute the displacements of vetices of SMPL model
    # P2Pdiff = meshDifference_pointTopoint( Tensor(np.array(SMPLvert)).to(ondevice), Tensor(dstVertices).to(ondevice) )
    # P2Fdiff = meshDifference_pointTofacet( Tensor(np.array(SMPLvert)).to(ondevice), Tensor(dstTriangulars).to(ondevice) )
    P2PNormDiff = computeNormalGuidedDiff( Tensor(np.array(SMPLvert)).to(ondevice), 
                                           Tensor(dstTriangulars).to(ondevice),
                                           maxDisp = dispThreshold,
                                           normalKNN = numNearestNeighbor)
    
    # remove displacements on unclothed parts, e.g. face, hands, foots
    if onlyCloth:
        for part, color in _neglectParts_.items():
            mask = (segmentations[:,0] == color[0])* \
                   (segmentations[:,1] == color[1])* \
                   (segmentations[:,2] == color[2])
            P2PNormDiff[mask] = 0
    
    # verify the displacements greater than the threshold by their triplets.
    if NNfiltering:
        meshDisplacementFilter(P2PNormDiff, Tensor(smplMesh).to(ondevice), threshold=filterThres)
        
    # save as ground truth displacement 
    if save:
        savePath = pjn(path_objectFolder, 'displacement/')
        if not exists(savePath):
            try:
                makedirs(savePath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        # with open( pjn(savePath, 'p2p_displacements.npy' ), 'wb' ) as f:
        #     np.save(f, P2Pdiff)
        # with open( pjn(savePath, 'p2f_displacements.npy' ), 'wb' ) as f:
        #     np.save(f, P2Fdiff)
        with open( pjn(savePath, 'normal_guided_displacements.npy' ), 'wb' ) as f:
            np.save(f, P2PNormDiff)


if __name__ == "__main__":
    
    path_object = '../../datasets/SampleDateset/125611487366942'
    computeDisplacement(path_object, 
                        '../../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                        device='cpu', dispThreshold = 0.1, numNearestNeighbor = 10)
    
    visDisplacement(path_object, '../../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', visMesh = True)
