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

import yaml
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


class meshDifferentiator():
    '''
    This class differentiate the registered scan and the SMPL model in the 
    given folder to compute the displacements of vertices.
    
    Data
    -------
    self.save : bool, optional
        Whether to save the displacements as .npy file.
    self.device : str, optional
        The device where the computation will take place.
    self.path_SMPLmodel : str
        Path to the SMPL models  file.
    self.num_separation : int
        In order to make memory footprint feasible, we separate the point
        cloud to the given batches(separatiions). 
        
    self.max_displacement : float 
        The maximum acceptable displacements of each vertices.
    self.numNearestNeighbor : int
        The number of nearest neigbor to estimate normal for vertices.
    self.onlyCloth: bool 
        Whether only computes the displacements of clothes, i.e. remove the 
        displacements of unclothed parts.
    self.filterThres : float, optional
        The threshold to specify suspects. 
    self.diffThreshold : float, optional
        The threshold to confirm outliers.
        
    '''
    
    def __init__(self, cfg: dict):
        '''
        Initialize the mesh displacement class with the given config dict.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary.

        '''
        # basic settings
        self.save = cfg['save_displacement']
        self.device = cfg['meshdiff_device']
        self.path_SMPLmodel = cfg['smplModel'] 
        self.num_separation = cfg['number_separation']
        self.enable_plots   = cfg['show_intermediateMeshes']
        
        # settings to upsample the SMPL mesh
        self.enable_meshdvd = cfg['divide_meshToSubmeshes']
        self.meshdvd_iters  = cfg['submeshesDivision_iters']
        self.meshdvd_algor  = cfg['submeshesDivision_alg']
        
        # settings to compute displacements
        self.max_displacement   = cfg['maximum_displacements']
        self.numNearestNeighbor = cfg['number_nearest_neighbor']
        
        # settings to filter the displacements
        self.onlyCloth = cfg['only_reconstruct_cloth']
        self.NNfiltering = cfg['enable_neighborsfilter']
        self.filterThres = cfg['thresh_neighborsfilter']
        self.diffThreshold = cfg['thresh_differ_NNfilter']
        self.fullyNNfilter = cfg['fully_neighborsfilter'] 
        self.NNfilterIters = cfg['neighborsfilter_iters']
        
        assert isfile(self.path_SMPLmodel), 'SMPL model not found.'
        assert self.device in ('cpu', 'cuda'), 'device = %s is not supported.'%self.device
        assert self.meshdvd_algor in ('simple', 'loop'), 'only support simple and loop upsampling method.'
        

    def meshUpsampleO3D(self, vertIn: Tensor, tripletsIn: Tensor, method: str,
                        numIters: int = 1, visMesh: bool = False) -> tuple:
    
        mesh_in = o3d.geometry.TriangleMesh()
        mesh_in.vertices = o3d.utility.Vector3dVector(vertIn)
        mesh_in.triangles= o3d.utility.Vector3iVector(tripletsIn)   
        mesh_in.compute_vertex_normals()
        
        if method == 'simple':
            mesh_dvd = mesh_in.subdivide_midpoint(number_of_iterations=numIters)
        elif method == 'loop':
            mesh_dvd = mesh_in.subdivide_loop(number_of_iterations=numIters)
        
        if visMesh:
            print('Upsample algorithm: %s \nmeshin(right) and the upsampled meshout (left)'%method)
            mesh_in.vertices = o3d.utility.Vector3dVector(vertIn + Tensor([0.6,0,0]))
            o3d.visualization.draw_geometries([mesh_dvd, mesh_in])
        
        vertout = Tensor(np.asarray(mesh_dvd.vertices))
        tripout = Tensor(np.asarray(mesh_dvd.triangles))
        
        return (vertout, tripout)
        

    def meshFilterO3D(self, displacements: Tensor, bodyVert: Tensor, triplets,
                      numiters: int = 1, visMesh: bool = False) -> Tensor:
        '''
        This function uses open3d to filter the given displacements. The
        Taubin filter is used since it can keep most details comparing to 
        the other 2 filters.

        Parameters
        ----------
        displacements : Tensor
            The displacements to be filtered.
        bodyVert : Tensor
            The body vertices.
        triplets : TYPE
            The triplets to form the body mesh.
        numiters : int, optional
            Numbe of iterations to conduct the filter.
            The default is 1.
        visMesh : bool, optional
            Whether to show the mesh after and before filting.
            The default is False.

        Returns
        -------
        Tensor
            The filtered displacements.

        '''
        assert displacements.shape == bodyVert.shape, \
               'dispalcements and bodyMesh should have the same size.'
        
        # prepare data
        dressedPC = displacements + bodyVert
        
        mesh_in = o3d.geometry.TriangleMesh()
        mesh_in.triangles= o3d.utility.Vector3iVector(triplets)   
        mesh_in.vertices = o3d.utility.Vector3dVector(dressedPC)
        
        # conduct filtering
        mesh_outTau = mesh_in.filter_smooth_taubin(number_of_iterations=numiters)        
         
        # compute the filtered displacements
        filteredDisplacements = Tensor(np.asarray(mesh_outTau.vertices)) - bodyVert 
        
        if visMesh:
            mesh_in.vertices = o3d.utility.Vector3dVector(dressedPC + Tensor([0.6,0,0]))
            mesh_in.compute_vertex_normals()
            mesh_outTau.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh_in, mesh_outTau])
            
            residual = (filteredDisplacements - displacements).norm(dim = 1).abs()
            print('max smoothening: %f \navg smoothening: %f \nmed smoothening: %f'% 
                  (residual.max(), residual.mean(), residual.median()))
        
        return filteredDisplacements


    def meshDisplacementFilter(self, displacements: Tensor, triplets: Tensor,
                               bodyVert: Tensor = None, numIters: int = None,
                               visMesh: bool = False):
        '''
        This function filters the suspects of outliers of the given displacements
        by the intialized thresholds.
        
        The self.filterThres specifies the suspects. Then the displacements of 
        the NN of suspectes are collected. If the difference between the 
        displacements of suspects and the median of their NN is greater than 
        the self.diffThreshold, the displacements of the suspects will be 
        replaced by the median.
    
        Ideally, it would be better to filter all points instead of focusing on 
        several obvious outliers, but full filtering would take time/memory, so 
        currently, we suggest not to do full filtering.
        
        Parameters
        ----------
        displacements : Tensor
            The displacements to be filtered, [Nx3].
        triplets : Tensor
            The mesh of the body, [Nx3].
        bodyVert : Tensor
            The body vertices, [Nx3].
            The default is None.
        numiters : int, optional
            Numbe of iterations to conduct open3d filter.
            The default is None.
    
        Returns
        -------
        displacements : TYPE
            The filtered displacements.
    
        '''
        
        # conduct NN filter on the whole point cloud using open3d
        if self.fullyNNfilter:
            assert bodyVert is not None and numIters is not None,  \
                   'To conduct full filter, body vertice and numIters have to be given.'
                   
            displacements = self.meshFilterO3D(displacements, bodyVert, 
                                               triplets, numIters, visMesh)
        
        # only conduct NN filter on suspects
        else:
            suspects = torch.nonzero(displacements.norm(dim=1) > self.filterThres, as_tuple=False)
            for pointInd in suspects:
                NNtriplets = torch.nonzero((triplets == pointInd).sum(dim=1), as_tuple=False)
                NNpointsId = triplets[NNtriplets].unique().long()
                vals, inds = displacements[NNpointsId].norm(dim=1).sort()
        
                medianInd  = floor(vals.shape[0]/2) - 1
                if (displacements[pointInd].norm() - vals[medianInd] ).abs() > self.diffThreshold:
                    displacements[pointInd] = displacements[ NNpointsId[inds[medianInd]] ]

        return displacements


    def computeNormalGuidedDiff(self, srcVertices: Tensor, dstMesh: Tensor) -> Tensor: 
        '''
        This function computes the displacements of vertices along their normal
        vectors estimated by the open3d package.
    
        Parameters
        ----------
        srcVertices : Tensor
            The SMPL vertices, [6890x3].
        dstMesh : Tensor
            The mesh of the registered scan, [Nx9].
    
        Returns
        -------
        Tensor
            The displacements of the given pointcloud to the given mesh.
    
        '''
        assert srcVertices.shape[1] == 3 and dstMesh.shape[1] == 9, \
            "src Vertices shape should be Nx3; dstMesh should be Nx9."
        assert (srcVertices.shape[0]/self.num_separation).is_integer(), \
            "vertices must divide batches to get an integer."
        
        KNNPara = o3d.geometry.KDTreeSearchParamKNN(self.numNearestNeighbor)
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
        segLength = int(srcSize/self.num_separation)
        for segCnt in range(self.num_separation):
            
            # get the subset of vertices and normal vectors
            tempVertices = srcVertices[ segCnt*segLength : (segCnt+1)*segLength, : ]
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
            # are far away from the source points(SMPL vertices)
            intersection_mask = (b1 > 0).logical_and(b2 > 0)      \
                                        .logical_and(b1+b2 < 1)   \
                                        .logical_and(t.abs() < self.max_displacement)
            
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
            indice = indice.unique(dim = 0)
            distance = t[indice[:,0], indice[:,1]] 
            normals  = tempNormalVec[indice[:,0]]
            displace = distance[:,None]*normals
            
            # fill data
            partialDisp = torch.zeros(segLength, 3)
            partialDisp[indice[:,0],:] = displace
            fullDisps.append( partialDisp )
        
        return torch.cat(fullDisps, dim = 0)
    

    def computeDiffO3d(self, srcPointCloud: np.array, dstPointCloud: np.array):
        '''
        This function computes the point-to-point difference between two 
        meshes, in the manner of dstVetices(NN) - srcVertices. We use open3d 
        to do this task as it is faster and manages memory better. 
        
        The output is exactly the same as we do it manmually, resErr = 10e-8.
                
        Parameters
        ----------
        srcVertices : Tensor
            The vertices of the source mesh which we would like to estimate 
            the displacement.
        dstVertices : Tensor
            The vertices of the destination mesh from which we could like to 
            estimate the displacement.
        Returns
        -------
        Tensor
            The point-to-point displacement of srcVertices to fit dstVetices.
            
        '''
        srcO3d = o3d.geometry.PointCloud()
        srcO3d.points = o3d.utility.Vector3dVector(srcPointCloud)
        
        dstO3d = o3d.geometry.PointCloud()
        dstO3d.points = o3d.utility.Vector3dVector(dstPointCloud)
        
        displacements = np.asarray(srcO3d.compute_point_cloud_distance(dstO3d))
        
        return Tensor(displacements)
    
    
    def computeDisplacement(self, path_objectFolder: str):
        '''
        This function read the given object and SMPL parameter to compute the
        displacement.
    
        Parameters
        ----------
        path_objectFolder : str
            Path to the folder containing MGN objects.
    
        Returns
        -------
        None.
    
        '''
        ondevice = ('cpu', self.device) [torch.cuda.is_available()]
    
        # prepare path to files
        objfile = pjn(path_objectFolder, 'smpl_registered.obj')
        regfile = pjn(path_objectFolder, 'registration.pkl')    # smpl model params
        segfile = pjn(path_objectFolder, 'segmentation.png')    # clothes segmentation 
        smplObj = pjn('/'.join(self.path_SMPLmodel.split('/')[:-1]), 'text_uv_coor_smpl.obj')
    
        # load registered mesh
        dstVertices, Triangle, _, _ = read_Obj(objfile)
        dstTriangulars = Tensor(dstVertices[Triangle.flatten()].reshape([-1, 9])).to(ondevice)
    
        # load segmentaiton and propagate to meshes
        segmentations  = np.array(Image.open(segfile))
        smplVert, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(smplObj)
        segmentations  = interpolateTexture(smplVert.shape, smplMesh, 
                                            segmentations, smpl_text_uv_coord, 
                                            smpl_text_uv_mesh)
        smplMesh = Tensor(smplMesh).to(ondevice)
        # visualizePointCloud(smplVert, segmentations)
    
        # load SMPL parameters and model, transform vertices and joints.
        registration = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
        SMPLmodel = load_model(self.path_SMPLmodel)
        SMPLmodel.pose[:]  = registration['pose']
        SMPLmodel.betas[:] = registration['betas']
        [SMPLvert, joints] = _verts_core(SMPLmodel.pose, SMPLmodel.v_posed, SMPLmodel.J,  \
                                      SMPLmodel.weights, SMPLmodel.kintree_table, want_Jtr=True, xp=ch)
        joints = joints + registration['trans']
        SMPLvert = Tensor(np.array(SMPLvert + registration['trans'])).to(ondevice)
    
        if self.enable_meshdvd:
            SMPLvert, smplMesh = self.meshUpsampleO3D(
                                    SMPLvert, smplMesh, self.meshdvd_algor, 
                                    self.meshdvd_iters, self.enable_plots)
    
        # compute the displacements of vetices of SMPL model
        P2PNormDiff = self.computeNormalGuidedDiff( SMPLvert, dstTriangulars )
        
        # verify the displacements greater than the threshold by their triplets.
        if self.NNfiltering:
            P2PNormDiff = self.meshDisplacementFilter(P2PNormDiff, smplMesh,
                                SMPLvert, self.NNfilterIters, self.enable_plots)
        
        # remove displacements on unclothed parts, e.g. face, hands, foots
        if self.onlyCloth:
            for part, color in _neglectParts_.items():
                mask = (segmentations[:,0] == color[0])* \
                       (segmentations[:,1] == color[1])* \
                       (segmentations[:,2] == color[2])
                P2PNormDiff[mask] = 0
                    
        # save as ground truth displacement 
        if self.save:
            savePath = pjn(path_objectFolder, 'displacement/')
            if not exists(savePath):
                try:
                    makedirs(savePath)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise        
            print('Saving dispalcements of: ', path_objectFolder)
            with open( pjn(savePath, 'normal_guided_displacements.npy' ), 'wb' ) as f:
                np.save(f, P2PNormDiff)
                

if __name__ == "__main__":
    
    path_object = '../../datasets/SampleDateset/125611487366942'
    
    with open('../dataset/MGN_render_cfg.yaml') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
        print('\nRendering configurations:\n', cfgs)   
    
    mesh_differ = meshDifferentiator(cfgs)
    mesh_differ.computeDisplacement(path_object)
    
    visDisplacement(path_object, cfgs['smplModel'], visMesh = True)
