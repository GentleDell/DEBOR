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
from os.path import join as pjn, isfile, exists, isdir
import pickle
import errno

import torch
from torch import Tensor
import open3d as o3d
import numpy as np
import chumpy as ch

from third_party.smpl_webuser.serialization import load_model
from third_party.smpl_webuser.lbs import verts_core as _verts_core 

def read_Obj(path):
    '''  
    See definition of Wavefront .obj file.
    
    'v'  represents the xyz coordinate of a vertex.
    'vt' represents the uv coordinate of a texture on the texture image.
    'f'  represents polygonal face element, example:
                  f  v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
         v1v2v3 are 3 indices to the 'v' domain to find the 3 points defining
         a triangular facet;
         vt1vt2vt3 are 3 indices to the 'vt' domain to fine the 3 uv coord 
         defining the color of the 3 vertices.
    
    '''
    vertices, texture_uv_coor, tri_list_uv, tri_list_vertex = [], [], [], []
    file = open(path, 'r');
    
    for line in file:
        
        tempData = line[:-1].replace('/', ' ').split(' ')
        
        if tempData[0] == 'v':    # vertices
            vertices.append( [float(data) for data in tempData[1:]] )
        if tempData[0] =='vt':    # texture coordinates of vertices
            texture_uv_coor.append( [float(data) for data in tempData[1:]] )
        if tempData[0] == 'f':    # triangular facets
            indicesNum = [float(data) for data in tempData[1:] if data != '']
            tri_list_vertex.append( [indicesNum[0], indicesNum[3], indicesNum[6]] )
            tri_list_uv.append( [indicesNum[1], indicesNum[4], indicesNum[7]] )
    
    vertices        = np.array(vertices) 
    texture_uv_coor = np.array(texture_uv_coor)
    tri_list_vertex = np.array(tri_list_vertex).astype('int') 
    tri_list_uv     = np.array(tri_list_uv).astype('int') 
    
    # index in np and python start from 0
    if tri_list_vertex.min() == 1:
        tri_list_vertex -= 1
    if tri_list_uv.min() == 1:
        tri_list_uv -= 1
      
    return vertices, tri_list_vertex, texture_uv_coor, tri_list_uv


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
                          save: bool = True, device: str = 'cuda'):
    '''
    This function read the given object and SMPL parameter to compute the
    displacement.

    Parameters
    ----------
    path_objectFolder : str
        Path to the folder containing MGN objects.
    path_SMPLmodel : str
        Path to the SMPL models  file.
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
    
    # load registered mesh
    dstVertices, Triangle, _, _ = read_Obj(objfile)
    dstTriangulars = dstVertices[Triangle.flatten()].reshape([-1, 9])
    
    # load SMPL parameters and model, transform vertices and joints.
    registration = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
    SMPLmodel = load_model(path_SMPLmodel)
    SMPLmodel.pose[:]  = registration['pose']
    SMPLmodel.betas[:] = registration['betas']

    [SMPLvert, Jtr] = _verts_core(SMPLmodel.pose, SMPLmodel.v_posed, SMPLmodel.J,  \
                                  SMPLmodel.weights, SMPLmodel.kintree_table, want_Jtr=True, xp=ch)
    Jtr = Jtr + registration['trans']
    SMPLvert = SMPLvert + registration['trans']
    
    # compute the displacements of vetices of SMPL model
    P2Pdiff = meshDifference_pointTopoint( Tensor(np.array(SMPLvert)).to(ondevice), Tensor(dstVertices).to(ondevice) )
    P2Fdiff = meshDifference_pointTofacet( Tensor(np.array(SMPLvert)).to(ondevice), Tensor(dstTriangulars).to(ondevice) )
    
    # save as ground truth displacement 
    if save:
        savePath = pjn(path_objectFolder, 'displacement/')
        if not exists(savePath):
            try:
                makedirs(savePath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open( pjn(savePath, 'point_to_point_displacements.npy' ), 'wb' ) as f:
            np.save(f, P2Pdiff)
        with open( pjn(savePath, 'point_to_facet_displacements.npy' ), 'wb' ) as f:
            np.save(f, P2Fdiff)
    

def visDisplacement(path_object: str, path_SMPLmodel: str, vis: bool = True,
                    meshDistance: float = 0.5, save: bool = True ):
    
    assert isfile(path_SMPLmodel), 'SMPL model not found.'
    assert isdir(path_object), 'the path to object folder is invalid.'
    
    # read registered mesh
    objfile = pjn(path_object, 'smpl_registered.obj')
    body_vertices, body_mesh, _, _ = read_Obj(objfile)
    
    # read SMPL model
    regfile = pjn(path_object, 'registration.pkl')    # smpl model params
    smplpara = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
    SMPLmodel = load_model(path_SMPLmodel)
    SMPLmodel.pose[:]  = smplpara['pose']
    SMPLmodel.betas[:] = smplpara['betas']
    [SMPLvert, Jtr] = _verts_core(SMPLmodel.pose, SMPLmodel.v_posed, SMPLmodel.J,  \
                                  SMPLmodel.weights, SMPLmodel.kintree_table, want_Jtr=True, xp=ch)
    Jtr = Jtr + smplpara['trans']
    SMPLvert = np.array(SMPLvert) + smplpara['trans'] + np.array([meshDistance, 0, 0])

    # read displacements 
    p2pfile = pjn(path_object, 'displacement/point_to_point_displacements.npy')
    p2ffile = pjn(path_object, 'displacement/point_to_facet_displacements.npy')
    p2pDisp = np.load(p2pfile)
    p2fDisp = np.load(p2ffile)
    p2pComp = SMPLvert + p2pDisp - 2*np.array([meshDistance, 0, 0])
    p2fComp = SMPLvert + p2fDisp - 3*np.array([meshDistance, 0, 0])
    
    # create meshes for visualization
    meshregister = o3d.geometry.TriangleMesh()
    meshregister.vertices = o3d.utility.Vector3dVector(body_vertices)
    meshregister.triangles= o3d.utility.Vector3iVector(body_mesh)
    
    meshSMPL = o3d.geometry.TriangleMesh()
    meshSMPL.vertices = o3d.utility.Vector3dVector(SMPLvert)
    meshSMPL.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    meshSMPLp2p = o3d.geometry.TriangleMesh()
    meshSMPLp2p.vertices = o3d.utility.Vector3dVector(p2pComp)
    meshSMPLp2p.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    meshSMPLp2f = o3d.geometry.TriangleMesh()
    meshSMPLp2f.vertices = o3d.utility.Vector3dVector(p2fComp)
    meshSMPLp2f.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0.5])
    o3d.visualization.draw_geometries([meshregister, meshSMPL, meshSMPLp2p, meshSMPLp2f, mesh_frame])

    if save:
        pathDisplacement = pjn(path_object, 'displacement/')
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "registered.obj"), meshregister)
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLposed.obj") , meshSMPL)
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLp2pCom.obj"), meshSMPLp2p)
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLp2fCom.obj"), meshSMPLp2f)

if __name__ == "__main__":
    
    path_object = '../../datasets/SampleDateset/125611487366942'
    # computeDisplacement(path_object, 
    #                       '../../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    #                       device='cpu')
    
    visDisplacement(path_object, '../../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', )
