#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:11:39 2020

@author: zhantao
"""
import sys
from os.path import join as isfile, abspath
if abspath('../') not in sys.path:
    sys.path.append(abspath('../'))
    sys.path.append(abspath('../dataset'))
    sys.path.append(abspath('../dataset/MGN_helper'))
    
import torch
from torch import Tensor
import open3d as o3d
import numpy as np
from numpy import array
import chumpy as ch
from psbody.mesh import Mesh

from MGN_helper.utils.smpl_paths import SmplPaths
from MGN_helper.lib.ch_smpl import Smpl
from third_party.smpl_webuser.serialization import load_model
from third_party.smpl_webuser.lbs import verts_core as _verts_core 


def create_smplD_psbody(SMPL :Smpl, offsets_t :array, pose :array, betas :array, 
                        trans :array, rtnMesh :bool=True):
    '''
    This function create a SMPLD body model and mesh from the given data.

    Parameters
    ----------
    SMPL : Smpl
        SMPLD model. If None, create one.
    offsets_t : array
        The per-vertex offsets in t-pose representing garments.
    pose : array
        The axis angle of 23+1 joints describing the rotation of each joint.
    betas : array
        The body shape coefficients.
    trans : array
        The global translation of the body.
    rtnMesh : bool, optional
        If return the created mesh in the output. The default is True.

    Returns
    -------
    out : TYPE
        List of SMPL model and corresponding mesh if required.

    '''
    
    if SMPL is None:
        # create a new smplD model, res depends on offsets_t
        dp = SmplPaths()
        is_hres = offsets_t.shape[0]>6890 if isinstance(offsets_t, array) else False
        SMPL = Smpl( dp.get_hres_smpl_model_data() if is_hres else dp.get_smpl_file() )
    
    SMPL.v_personal[:] = offsets_t
    SMPL.pose[:]  = pose
    SMPL.betas[:] = betas
    SMPL.trans[:] = trans
    
    out = [SMPL]
    
    if rtnMesh:
        body = Mesh(SMPL.r, SMPL.f)
        out.append(body)
    
    return out
    

def create_fullO3DMesh(vertices: Tensor, triangles: Tensor, 
                       texturePath: str = None, segmentationPath: str = None, 
                       triangle_UVs: Tensor = None, vertexcolor: Tensor = None,
                       use_text: bool=False, use_vertex_color: bool = False,
                       ) -> o3d.geometry.TriangleMesh:
    '''
    This function creates a open3d triangular mesh object with vertices, edges,
    textures, segmentations. 
    
    Parameters
    ----------
    vertices : Tensor
        The vertices of the mesh.
    triangles : Tensor
        The edges of the mesh.
    texturePath : str , optional 
        The path to the textures.
        The default is None.
    segmentationPath : str , optional 
        The path to the segmentation.
        The default is None.
    triangle_UVs : Tensor , optional 
        The UV coodiantes of all triangles, i.e. triangleUVs = uvs[meshUVInd].
        The default is None.
    vertexcolor : Tensor , optional 
        The color of vetices of the mesh to be created.
        The default is None.
    use_text : bool , optional 
        Whether to use texture for vis. It has higher priority than the 
        following use_vertex_color.
        The default is False
    use_vertex_color: bool , optional 
        Whether to use vertex colors for visualization.
        The default is False,

    Returns
    -------
    o3dMesh : o3d.geometry.TriangleMesh
        The created open3d triangle mesh object.

    '''
    # create mesh 
    o3dMesh = o3d.geometry.TriangleMesh()
    o3dMesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3dMesh.triangles= o3d.utility.Vector3iVector(triangles) 
    o3dMesh.compute_vertex_normals()     

    # if use texure image for visualization, set textures and prepare the 
    # material ids for vertices
    if use_text:
        assert triangle_UVs is not None \
            and texturePath is not None \
            and segmentationPath is not None, \
            'triangle_UVs, texture image and segmentation have to be set.'
        
        textureImage = o3d.geometry.Image( o3d.io.read_image(texturePath) )
        segmentImage = o3d.geometry.Image( o3d.io.read_image(segmentationPath) )
        texture_Idx  = o3d.utility.IntVector(torch.zeros(triangles.shape[0]).int())    # [numTriangle, 1]
        
        o3dMesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_UVs)
        o3dMesh.textures = [o3d.geometry.Image(textureImage), 
                            o3d.geometry.Image(segmentImage)]
        o3dMesh.triangle_material_ids = texture_Idx
        
    # else if use vertex color for visualization, set color for each vertex
    elif use_vertex_color and vertexcolor is not None:
        o3dMesh.vertex_colors = o3d.utility.Vector3dVector(vertexcolor/vertexcolor.max())
            
    return o3dMesh


def generateSMPLmesh(path_toSMPL: str, pose: array, beta: array, trans: array,
                     asTensor: bool = True, device: str = 'cpu' ) -> tuple:
    '''
    This function reads the SMPL model in the given path and then poses and 
    reshapes the mesh. 

    Parameters
    ----------
    path_toSMPL : str
        Path to the SMPL model.
    pose : array
        The poses of joints.
    beta : array
        The parameters of body shape.
    trans : array
        The global translation of the mesh.
    asTensor : bool, optional
        Output the mesh as Tensor. 
        The default is True.
    device : str, optional
        The device to store the mesh, only vaild when asTensor is Ture. 
        The default is 'cpu'.

    Returns
    -------
    tuple
        Tuple of SMPL mesh and SMPL joints.

    '''
    assert isfile(path_toSMPL), 'SMPL model not found.'
    assert device in ('cpu', 'cuda'), 'device = %s is not supported.'%device
    
    # read SMPL model and set SMPL parameters
    SMPLmodel = load_model(path_toSMPL)
    SMPLmodel.pose[:]  = pose
    SMPLmodel.betas[:] = beta
    
    # blender the mesh 
    [SMPLvert, joints] = _verts_core(SMPLmodel.pose, SMPLmodel.v_posed, SMPLmodel.J,  \
                                  SMPLmodel.weights, SMPLmodel.kintree_table, want_Jtr=True, xp=ch)
    joints = np.array(joints + trans)
    SMPLvert = np.array(SMPLvert + trans)

    if asTensor:
        joints   = Tensor(joints).to(device)
        SMPLvert = Tensor(SMPLvert).to(device)
        
    return (SMPLvert, joints)
