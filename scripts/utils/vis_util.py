#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:16:11 2020

@author: zhantao
"""
import pickle as pickle
from os.path import join as pjn, isfile, isdir, abspath
import sys
if abspath('../') not in sys.path:
    sys.path.append(abspath('../'))
    sys.path.append(abspath('../third_party/smpl_webuser'))
if abspath('../dataset') not in sys.path:
    sys.path.append(abspath('../dataset'))
if abspath('../dataset/MGN_helper') not in sys.path:
    sys.path.append(abspath('../dataset/MGN_helper'))

import numpy as np
from numpy import array

from psbody.mesh import Mesh, MeshViewers
from MGN_helper.utils.smpl_paths import SmplPaths
from MGN_helper.lib.ch_smpl import Smpl
if __name__ == "__main__":
    from mesh_util import create_smplD_psbody
else:
    from .mesh_util import create_smplD_psbody


def read_Obj(path):
    '''  
    See definition of Wavefront .obj file.
    
    'v'  represents the xyz coordinate of a vertex.
    'vt' represents the uv coordinate of a texture on the texture image.
    'f'  represents polygonal face element, example:
                  f  v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
         v1v2v3 are 3 indices to the 'v' domain to find the 3 points defining
         a triangular facet;
         vt1vt2vt3 are 3 indices to the 'vt' domain to find the 3 uv coord 
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


def vis_subjectFromPath(path_object: str, path_SMPLmodel: str, 
                        path_bodyMesh: str, is_hres = False):
    '''
    This function visualize object file in the given path.

    Parameters
    ----------
    path_object : str
        Path to the folder constaining the object.
    path_SMPLmodel : str
        Path to the SMPL model .pkl file.

    Returns
    -------
    None.

    '''
    assert isfile(path_SMPLmodel), 'SMPL model not found.'
    assert isdir(path_object), 'the path to object folder is invalid.'
    assert isfile(path_bodyMesh) , 'reference mesh not found.'
    assert ('smpl_registered' in path_bodyMesh) == is_hres, \
        'reference mesh has different resolution from that specified by is_hres.'
    
    dp = SmplPaths()
    SMPL = Smpl( dp.get_hres_smpl_model_data() if is_hres else dp.get_smpl_file() )
    
    # read pose
    regfile = pjn(path_object, 'registration.pkl')    # smpl model params
    smplpara = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
    
    # read offsets
    offsets_t = np.load( pjn(path_object, 'gt_offsets/offsets_%s.npy')%\
                      (['std', 'hres'][is_hres]) )
    
    # prepare texture
    tex_path = pjn(path_object, 'registered_tex.jpg')
    
    vis_subjectFromData_psbody(
        SMPL, 
        offsets_t, 
        smplpara['pose'],
        smplpara['betas'], 
        smplpara['trans'],
        path_bodyMesh, 
        tex_path, 
        is_hres)


def vis_subjectFromData_psbody(
        SMPL:Smpl, offsets_t:array, pose:array, betas:array, 
        trans: array, refMesh_path: str, tex_path: str, 
        is_hres: bool):

    if SMPL is None:
        dp = SmplPaths()
        SMPL = Smpl( dp.get_hres_smpl_model_data() if is_hres else dp.get_smpl_file() )
    
    # t-pose smpl body mesh
    body_t = Mesh(SMPL.r + offsets_t, SMPL.f)
    if refMesh_path is not None and tex_path is not None:
        _, _, t, ft = read_Obj(refMesh_path)    
        body_t.vt = t
        body_t.ft = ft
        body_t.set_texture_image(tex_path)

    # posed smplD body mesh
    SMPL, body_p = create_smplD_psbody(SMPL, offsets_t, pose, betas, trans)
    if refMesh_path is not None and tex_path is not None:
        body_p.vt = t
        body_p.ft = ft
        body_p.set_texture_image(tex_path)
        
    mvs = MeshViewers((1,2))
    mvs[0][0].set_static_meshes([ body_t ])
    mvs[0][1].set_static_meshes([ body_p ])
    

if __name__ == "__main__":
    
    path_object = '../../datasets/Multi-Garment_dataset/125611520103063_coat_094_pants_096_pose_096'
    path_SMPLpkl= '../../body_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    
    vis_subjectFromPath(
        path_object = path_object, 
        path_SMPLmodel = path_SMPLpkl,
        path_bodyMesh = '../../body_model/text_uv_coor_smpl.obj',    # std uses: '../../body_model/text_uv_coor_smpl.obj'
        is_hres = False                                              # hres uses: pjn(path_object, 'smpl_registered.obj')
        )
