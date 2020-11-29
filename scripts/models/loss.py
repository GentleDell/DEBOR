#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:05:16 2020

@author: zhantao
"""
from os.path import abspath
import sys
if abspath('./') not in sys.path:
    sys.path.append(abspath('./'))
    sys.path.append(abspath('./third_party/kaolin'))
    
import torch


def normal_loss(bs, pred_mesh, gt_mesh, body_mesh, num_faces):
    body_normals = []
    gt_normals = []
    pred_normals = []
    for i in range(bs):
        b_normal = body_mesh[i].compute_face_normals()
        body_normals.append(b_normal)
        gt_nromal = gt_mesh[i].compute_face_normals()
        pred_normal = pred_mesh[i].compute_face_normals()
        gt_normals.append(gt_nromal)
        pred_normals.append(pred_normal)

    body_normals = torch.stack(body_normals)
    gt_normals = torch.stack(gt_normals)
    pred_normals = torch.stack(pred_normals)
    loss_norm = torch.sum(torch.sum((1 - gt_normals) * pred_normals, dim=2).abs()) / (bs * num_faces)

    # in CAPE
    # cos = tf.reduce_sum(tf.multiply(normals_pred, normals_gt), axis=-1)
    # cos_abs = tf.abs(cos)
    # normal_loss = 1 - cos_abs

    return loss_norm, body_normals, pred_normals


def edge_loss_calc(pred, gt, vpe):
    '''
    Calculate edge loss measured by difference in the length
    args:
        pred: prediction, [batch size, num_verts (6890), 3]
        gt: ground truth, [batch size, num_verts (6890), 3]
        vpe: SMPL vertex-edges correspondence table, [20664, 2]
    returns:
        edge_obj, an array of size [batch_size, 20664], each element of the second dimension is the
        length of the difference vector between corresponding edges in GT and in pred
    '''
    # get vectors of all edges, have size (batch_size, 20664, 3)
    edges_vec = lambda x: tf.gather(x,vpe[:,0],axis=1) -  tf.gather(x,vpe[:,1],axis=1)
    edge_diff = edges_vec(pred) -edges_vec(gt) # elwise diff between the set of edges in the gt and set of edges in pred
    edge_obj = tf.norm(edge_diff, ord='euclidean', axis=-1)

    return tf.reduce_mean(edge_obj)


def MS_DSSIM():
    '''
    Multi-Scale DisSImilarity Metrics
    tex2shape does not release the codes. But  we find a git repo for this
    metrics. Add it later.
    
    '''
    raise NotImplementedError('to be implemented')
    
    
    