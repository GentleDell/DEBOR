#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:44:23 2020

@author: zhantao
"""

import torch
from torch import Tensor
import torch.nn as nn

from psbody.mesh import Mesh, MeshViewer
from psbody.mesh.visibility import visibility_compute


class cameraPerspective(nn.Module):
    '''
    Perspective camera object to preject points to camera plane.
        
    '''
    def __init__( self, smpl_obj: bool = True):
        super(cameraPerspective, self).__init__()
        
        self.SMPL_objcoord = smpl_obj
        self.R_vertx2blobj = torch.Tensor(
                          [[[1, 0, 0],
                            [0, 0,-1],
                            [0, 1, 0]]]).double()
        
    def vertices_visibility(self, vertices: Tensor, faces : Tensor, 
                            camtrans: Tensor):
        '''
        This function check the visibility of vertices for the given camera, 
        using the MPI mesh library. 
        
        The coordinate system is the same as .obj file which is different
        from ours and blender's.
        
        Process:
            1. The library create a AABB tree for the mesh.
            2. For each vertex of the mesh, it computes the ray pointing from 
            the vertex to the camera center.
            3. Given the ray and the mesh, it checks if the ray intersected 
            with the mesh (invisible) and return the results.
            
        It assumes that the given camera always points to the origin, so the
        input is camera translation only.
        
        It also support to add camera frame size as constrain and additional
        meshes as occulusion. (not used here)
            
        Parameters
        ----------
        vertices : Tensor
            Vertices of the mesh to be checked.
        faces : Tensor
            Faces of the mesh.
        camtrans : Tensor
            Camera translation of the camera to compute visibility.

        Returns
        -------
        visIndices : Tensor
            A mask vector indicating the visibility.

        '''
        assert camtrans.shape[1] == 3, 'camtrans must be Bx3 Tensor.'
        assert len(vertices.shape) == 2 and len(faces.shape) == 2
        
        # cast data type
        if vertices.requires_grad:
            vert = vertices.detech().cpu().double().numpy()
        else:
            vert = vertices.cpu().double().numpy()
            
        face = faces.cpu().numpy().astype('uint32')
        cam  = camtrans.cpu().double().numpy()
        
        # compute visibility
        visIndices, _ = visibility_compute(v=vert, f=face, cams=cam )
        
        # # debug
        # visIndices = visIndices.T.repeat(3, axis = 1)
        # my_mesh = Mesh(v=vert, f=face, vc=visIndices)
        # mvs = MeshViewer()
        # mvs.set_static_meshes([my_mesh])
        
        return visIndices
    
    def forward(self, fx, fy, cx, cy, rotation, translation, points, 
                faces, visibilityOn = True):
        '''
        It projects the given meshes to the given camera plane. If visibility 
        is enabled, it will project the visible points only

        Parameters
        ----------
        fx, fy, cx, cy : 
            camera intrinsics, [B,1,1]. 
        rotation, translation :
            camera extrinsics, [B,3,3].
        points : 
            vertices of the mesh, [B,6890,3].
        faces : 
            faces of the mesh, [B,13776,3].
        visibilityOn : bool, optional
            project only the visible points or not. The default is True.

        Returns
        -------
        pixels : list
            the projected points on image coordinate, [B,N,2], in {col, row}.
        visibility:
            the mask indicating vertices visibility, [B,6890].

        '''
        batch_size = points.shape[0]       
        
        # visibility check
        if visibilityOn:
            visibility = []
            for cnt in range(batch_size):
                trans = self.R_vertx2blobj[0].T@\
                            (-rotation[cnt].T)@\
                                translation[cnt].T    # convert trans to SMPL coord
                visIndices = self.vertices_visibility(
                    points[cnt], faces[cnt], trans[None])
                visibility.append(visIndices)
            visibility = torch.Tensor(visibility).permute(0,2,1)[:,:,0]
            
        # Convert coordinate system of points from SMPL model to ours.
        #     The vertices of the output of SMPL model (rXuYbZ), is different  
        #     from what we/blender use, rXfYuZ.
        if self.SMPL_objcoord: 
            R_vertx2blobj = \
                self.R_vertx2blobj.repeat_interleave(
                    batch_size, dim = 0).to(points.device)
                
            points = torch.einsum(
                'bji,bmi->bmj', R_vertx2blobj, points)
    
        with torch.no_grad():
            camera_mat = torch.zeros([batch_size, 3, 3], device=points.device).double()

            camera_mat[:, 0, 0] = fx[:,0,0]
            camera_mat[:, 1, 1] = fy[:,0,0]
            camera_mat[:, 0, 2] = cx[:,0,0]
            camera_mat[:, 1, 2] = cy[:,0,0]    
            camera_mat[:, 2, 2] = 1 

        # project vertices to image
        loc_points = torch.einsum(
            'bji,bmi->bmj',
             rotation, points) + translation

        front_para = torch.div(loc_points,
                               loc_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bmi,bji->bjm', camera_mat, front_para)[:,:,:2]
        
        # remove invisible vertices
        pixels = []
        for cnt in range(batch_size):
            if visibilityOn:
                pixels.append( img_points[cnt, visibility[cnt]==1] )    
            else:
                pixels.append( img_points[cnt] )
        
        return pixels, visibility
        