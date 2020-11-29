#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:16:11 2020

@author: zhantao
"""
import pickle as pickle
from os.path import join as pjn, isfile, isdir, abspath
import sys
if abspath('./') not in sys.path:
    sys.path.append(abspath('./'))
    sys.path.append(abspath('./third_party/smpl_webuser'))

import open3d as o3d
import chumpy as ch
import numpy as np
from numpy import array
from torch import Tensor
from scipy import interpolate

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


def visDisplacement(path_object: str, path_SMPLmodel: str, visMesh: bool,
                    meshDistance: float = 0.5, save: bool = True ):
    '''
    This function visualize object file in the given path.

    Parameters
    ----------
    path_object : str
        Path to the folder constaining the object.
    path_SMPLmodel : str
        Path to the SMPL model .pkl file.
    meshDistance : float, optional
        The distance between meshes. 
        The default is 0.5.
    save : bool, optional
        Whether to save the SMPL+displacements as obj file. 
        The default is True.

    Returns
    -------
    None.

    '''
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
    # p2pfile = pjn(path_object, 'displacement/p2p_displacements.npy')
    # p2ffile = pjn(path_object, 'displacement/p2f_displacements.npy')
    p2pNormfile = pjn(path_object, 'GroundTruth/normal_guided_displacements_oversample_OFF.npy')
    
    # p2pDisp = np.load(p2pfile)
    # p2fDisp = np.load(p2ffile)
    p2pNormDisp = np.load(p2pNormfile)
    
    # p2pComp = SMPLvert + p2pDisp - 2*np.array([meshDistance, 0, 0])
    # p2fComp = SMPLvert + p2fDisp - 3*np.array([meshDistance, 0, 0])
    p2pNormComp = SMPLvert + p2pNormDisp - 4*np.array([meshDistance, 0, 0])
    
    # create meshes for visualization
    meshregister = o3d.geometry.TriangleMesh()
    meshregister.vertices = o3d.utility.Vector3dVector(body_vertices)
    meshregister.triangles= o3d.utility.Vector3iVector(body_mesh)
    
    meshSMPL = o3d.geometry.TriangleMesh()
    meshSMPL.vertices = o3d.utility.Vector3dVector(SMPLvert)
    meshSMPL.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    # meshSMPLp2p = o3d.geometry.TriangleMesh()
    # meshSMPLp2p.vertices = o3d.utility.Vector3dVector(p2pComp)
    # meshSMPLp2p.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    # meshSMPLp2f = o3d.geometry.TriangleMesh()
    # meshSMPLp2f.vertices = o3d.utility.Vector3dVector(p2fComp)
    # meshSMPLp2f.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    meshSMPLp2pNorm = o3d.geometry.TriangleMesh()
    meshSMPLp2pNorm.vertices = o3d.utility.Vector3dVector(p2pNormComp)
    meshSMPLp2pNorm.triangles= o3d.utility.Vector3iVector(SMPLmodel.f)
    
    # visualize mesh to debug
    if visMesh:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0.5])
        o3d.visualization.draw_geometries([meshregister, meshSMPL, mesh_frame, meshSMPLp2pNorm])

    if save:
        pathDisplacement = pjn(path_object, 'GroundTruth/')
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "registered.obj"), meshregister)
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLposed.obj") , meshSMPL)
        # o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLp2pCom.obj"), meshSMPLp2p)
        # o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLp2fCom.obj"), meshSMPLp2f)
        o3d.io.write_triangle_mesh( pjn(pathDisplacement, "SMPLp2pNormCom.obj"), meshSMPLp2pNorm)
        

def visualizePointCloud(pointCloud: Tensor, texture: Tensor = None, 
                        normalKNN: int = 5):
    '''
    This function visualize the given point cloud by open3D package. The color
    is normal vector of vertices.

    Parameters
    ----------
    pointCloud : Tensor
        The point cloud to be visualized, [Nx3].
    texture : Tensor
        The texture of to be draw on the point cloud, [Nx3].
    normalKNN : int, optional
        The number of nearest neigbors to approximate the normal. 
        The default is 5.

    Returns
    -------
    None.

    '''    
    KNNPara = o3d.geometry.KDTreeSearchParamKNN(normalKNN)
    
    # use open3d to estimate normals
    PointCloud = o3d.geometry.PointCloud()
    PointCloud.points = o3d.utility.Vector3dVector(pointCloud)
    PointCloud.estimate_normals(search_param = KNNPara)
    
    if texture is not None:
        PointCloud.colors = o3d.utility.Vector3dVector(texture)
        
    # visualize the estimated normal on the mesh
    o3d.visualization.draw_geometries([PointCloud])
    

def interpolateTexture(verticeShape: tuple, mesh: array, texture: array, 
                       text_uv_coord: array, text_uv_mesh: array, 
                       algorithm: str = 'nearestNeighbors') -> array:
    '''
    This function interpolates the given texture for the given mesh. 

    Parameters
    ----------
    verticeShape : tuple
        The shape of the vertices of the mesh.
    mesh : array
        The edges of the mesh, [Nx3].
    texture : array
        The texture image of the mesh, [HxW].
    text_uv_coord : array
        The uv coordiate to get the color from the texture.
    text_uv_mesh : array
        Specifying the id of uv_coord of triplets.
    algorithm : str, optional
        Which interpolation algorithm to use. 
        The default is 'nearestNeighbors'.

    Returns
    -------
    array
        The color of the mesh.

    '''
    assert algorithm in ('nearestNeighbors', 'cubic'), "only support nearestNeighbors and cubic."
    
    UV_size = texture.shape[0]
    text_uv_coord[:, 1] = 1 - text_uv_coord[:, 1];
    x = np.linspace(0, 1, UV_size)
    y = np.linspace(0, 1, UV_size)
    meshColor = np.zeros(verticeShape)
    
    if algorithm == 'cubic':
        R = interpolate.interp2d(x, y, texture[:,:,0], kind='cubic')
        G = interpolate.interp2d(x, y, texture[:,:,1], kind='cubic')
        B = interpolate.interp2d(x, y, texture[:,:,2], kind='cubic')
        
        for indVt, induv in zip(mesh.flatten(), text_uv_mesh.flatten()):
            meshColor[indVt,:] = np.stack((R(text_uv_coord[induv,0], text_uv_coord[induv,1]),
                                           G(text_uv_coord[induv,0], text_uv_coord[induv,1]),
                                           B(text_uv_coord[induv,0], text_uv_coord[induv,1])), 
                                          axis = 1)
    elif algorithm == 'nearestNeighbors':
        xg, yg = np.meshgrid(x,y)
        coords = np.stack([xg.flatten(), yg.flatten()], axis = 1)
        interp = interpolate.NearestNDInterpolator(coords, texture.reshape(-1, 3))

        for indVt, induv in zip(mesh.flatten(), text_uv_mesh.flatten()):
            meshColor[indVt,:] = interp(text_uv_coord[induv,0], text_uv_coord[induv,1])
        
    return meshColor