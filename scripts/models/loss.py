#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:05:16 2020

@author: zhantao
"""
from os.path import abspath
import sys

if __name__ == '__main__':
    if abspath('../') not in sys.path:
        sys.path.append(abspath('../'))
    
import torch
import open3d as o3d
from pytorch3d.structures import Meshes


def normal_loss(pred_verts, gt_verts, triplets, device = 'cpu'):
    '''
    Compute the normal differences of triangles(facet) of the predicted verts
    and GT verts, through the pytorch3d package.

    Parameters
    ----------
    pred_verts : Tensor
        Predicted displacements + body vertices, [B,V,3].
    gt_verts : Tensor
        GT displacements + body vertices, [B,V,3].
    triplets : Tensor
        The triangles of the mesh, [B,C,3].

    Returns
    -------
    loss_face_normal : Tensor
        Normal difference.

    '''
    # prepare and cast data type
    batchSize = pred_verts.shape[0]
    num_verts = pred_verts.shape[1]
    num_faces = triplets.shape[1]
    
    # create pytorch3d tri mesh
    pred_mesh = Meshes(verts=pred_verts, faces=triplets)
    gt_mesh   = Meshes(verts=gt_verts, faces=triplets)
    
    # compute vertices and faces normals
    pred_mesh_verts_norm = pred_mesh.verts_normals_packed()
    pred_mesh_faces_norm = pred_mesh.faces_normals_packed()
    
    gt_mesh_verts_norm   = gt_mesh.verts_normals_packed()
    gt_mesh_faces_norm   = gt_mesh.faces_normals_packed()
    
    # Debug:
    #     residual error comparing to open3d is:
    #     1e-7 for verts, 3e-4 or faces, on average.
    # o3dMesh = o3d.geometry.TriangleMesh()
    # o3dMesh.vertices = o3d.utility.Vector3dVector(gt_verts[0].detach().cpu())
    # o3dMesh.triangles= o3d.utility.Vector3iVector(triplets[0].detach().cpu()) 
    # o3dMesh.compute_vertex_normals()     
    # o3dMesh.compute_triangle_normals()   
    # pred_vn = np.array(o3dMesh.vertex_normals)
    # pred_fn = np.array(o3dMesh.triangle_normals)
    # o3dMesh.vertex_colors = o3d.utility.Vector3dVector(pred_fn[-6890:] - 
    #                                     gt_mesh_faces_norm[13776-6890:13776]
    #                                    .detach().cpu().numpy())
    # o3d.visualization.draw_geometries([o3dMesh])
    
    # triangle verts' normal loss
    loss_vert_normal = torch.sum( 
        1 - torch.sum( gt_mesh_verts_norm*pred_mesh_verts_norm, dim=1
                     ).abs()) / (batchSize * num_verts)
    
    # triangle faces' normal loss
    loss_face_normal = torch.sum( 
        1 - torch.sum( gt_mesh_faces_norm*pred_mesh_faces_norm, dim=1
                     ).abs()) / (batchSize * num_faces)
    
    return loss_vert_normal, loss_face_normal


def edge_loss(pred_verts, gt_verts, vpe):
    '''
    Calculate edge loss measured by difference in the length. 
    
    Adapted from CAPE.
    
    args:
        pred_verts: prediction, [batch size, num_verts (6890), 3]
        gt_verts: ground truth, [batch size, num_verts (6890), 3]
        vpe: SMPL vertex-edges correspondence table, [20664, 2]
    returns:
        loss_edge: L2 norm loss of edge difference.
    '''
    # get vectors of all edges, have size (batch_size, 20664, 3)
    edges_pred = pred_verts[:, vpe[:,0]] - pred_verts[:, vpe[:,1]] 
    edges_gt   = gt_verts[:, vpe[:,0]] - gt_verts[:, vpe[:,1]] 
    
    edge_diff = edges_pred - edges_gt
    
    loss_edge = edge_diff.norm(dim = 2).mean(dim = 1).sum()/pred_verts.shape[0]

    return loss_edge


def MS_DSSIM():
    '''
    Multi-Scale DisSImilarity Metrics
    tex2shape does not release the codes. But  we find a git repo for this
    metrics. Add it later.
    
    '''
    raise NotImplementedError('to be implemented')
    
    
    