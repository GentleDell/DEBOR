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
        sys.path.append(abspath('../third_party/kaolin/'))
    
from kaolin.rep import TriangleMesh as tm
import torch
import open3d as o3d


def normal_loss(pred_verts, gt_verts, triplets):
    '''
    Compute the normal differences of triangles(facet) of the predicted verts
    and GT verts, through the Kaolin package.

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
    device = pred_verts.device
    batchSize = pred_verts.shape[0]
    num_faces = triplets.shape[0]
    triplets = triplets.type(torch.LongTensor).to(device)
    
    # create kaolin tri mesh
    pred_mesh = [tm.from_tensors(vertices=v,
                faces=triplets) for v in pred_verts]
    gt_mesh = [tm.from_tensors(vertices=v,
               faces=triplets) for v in gt_verts]
    
    # compute normals
    gt_face_normals, pred_face_normals = [], []
    for i in range(batchSize):
               
        # triangle facets' normals 
        #     comparing to open3d, residual diff is at 1e-4 level
        gt_nromal = gt_mesh[i].compute_face_normals()
        pred_normal = pred_mesh[i].compute_face_normals()
        gt_face_normals.append(gt_nromal)
        pred_face_normals.append(pred_normal)
    
    gt_face_normals = torch.stack(gt_face_normals)
    pred_face_normals = torch.stack(pred_face_normals)
    
    # triangle facets' normal loss
    loss_face_normal = torch.sum( 1 - torch.sum( gt_face_normals * pred_face_normals, dim=2).abs() ) \
                        / (batchSize * num_faces)
    
    return loss_face_normal


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
    
    
    