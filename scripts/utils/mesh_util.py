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
import torch
from torch import Tensor

import open3d as o3d
import numpy as np
from numpy import array
from scipy import interpolate
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
        Path to the SMPL models file.
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
        self.path_SMPLmodel = cfg['smplModel_neu'] 
        self.num_separation = cfg['number_separation']
        self.enable_plots   = cfg['show_intermediateMeshes']
        
        # settings to upsample the SMPL mesh
        self.enable_meshdvd = cfg['divide_meshToSubmeshes']
        self.meshdvd_iters  = cfg['submeshesDivision_iters']
        self.meshdvd_algor  = cfg['submeshesDivision_alg']
        self.meshdvd_thres  = cfg['textureGap_threshold']
        
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
        

    def meshUVcoordUpsample(self, mesh_orig: o3d.geometry.TriangleMesh,
                            mesh_dvd: o3d.geometry.TriangleMesh, 
                            ) -> o3d.geometry.TriangleMesh:
        '''
        This function upsamplea the triangle UVs of the given triangle mesh.
        
        We follow a simple ideal to interpolate the UV coords of newly sampled 
        points. For example, d is the upsampled midpoint of edge AB, then we
        pick UV_d = (UV_A + UV_B)/2.
        
        But this brings inconsistency to the upsampled textures since textures
        on the UV image is inconsistent. So, a correction is conducted to keep
        the color of [vertices] correct.

        Parameters
        ----------
        mesh_orig : o3d.geometry.TriangleMesh
            The unsampled original open3d triangle mesh.
        mesh_dvd : o3d.geometry.TriangleMesh
            The upsampled open3d triangle mesh, its UVs are to be sampled.

        Returns
        -------
        mesh_dvd : o3d.geometry.TriangleMesh
            The upsampled open3d triangle mesh and its UVs are sampled too.

        '''
        mesh_dvd.textures = mesh_orig.textures
        
        tempIndices = np.asarray(mesh_dvd.triangles)
        tempTriUVs  = np.asarray(mesh_orig.triangle_uvs)
        tempTriInds = np.asarray(mesh_orig.triangles).flatten()
        
        # get the UV coords of all vertices of the original mesh
        _, indices = np.unique(tempTriInds, return_index=True)
        origColorUV = tempTriUVs[indices]
        
        # the way that open3d upsamples triangle mid points
        # Triangle ABC: AB -> d, BC -> e, CA -> f; 1 triangle -> 4 triangle.
        # ABC => [Adf, dBe, eCf, def]
        A_, B_, C_ = tempIndices[::4,0], tempIndices[1::4,1], tempIndices[2::4,1]
        d_, e_, f_ = tempIndices[::4,1], tempIndices[1::4,2], tempIndices[ ::4,2]
        
        # mid points of triangle -> mid points on texture UV coordinate sys
        UV_A, UV_B, UV_C = origColorUV[A_], origColorUV[B_], origColorUV[C_]
        uv_d, uv_e, uv_f = (UV_A + UV_B)/2, (UV_B + UV_C)/2, (UV_C + UV_A)/2
        
        # correct texture gap
        invalid = np.linalg.norm(UV_A - UV_B, axis = 1) > self.meshdvd_thres
        uv_d[invalid] = UV_A[invalid] + 0*(UV_B[invalid] - UV_A[invalid])
        invalid = np.linalg.norm(UV_B - UV_C, axis = 1) > self.meshdvd_thres                   
        uv_e[invalid] = UV_B[invalid] + 0*(UV_C[invalid] - UV_B[invalid])
        invalid = np.linalg.norm(UV_C - UV_A, axis = 1) > self.meshdvd_thres
        uv_f[invalid] = UV_C[invalid] + 0*(UV_A[invalid] - UV_C[invalid])
        
        # orgnize the upsampled vertices and triangles
        allVertice = np.vstack([A_, B_, C_, d_, e_, f_]).flatten()
        allTextUVs = np.vstack([UV_A, UV_B, UV_C, uv_d, uv_e, uv_f])
        
        # get the UVs of all triangles of the upsampled mesh
        _, vertexInds = np.unique( allVertice, return_index = True)
        verticeTxtUVs = allTextUVs[vertexInds]
        triangularUVs = verticeTxtUVs[tempIndices.flatten()]
        
        # set material index for all vertices for visualization
        material_Idx  = o3d.utility.IntVector(torch.zeros(tempIndices.shape[0]).int())
        mesh_dvd.triangle_material_ids = material_Idx
        
        # set the triangle_uvs
        mesh_dvd.triangle_uvs = o3d.utility.Vector2dVector(triangularUVs)
        
        return mesh_dvd


    def meshVertexColorInterp(self, texture: array, vertUVs: array, algorithm: str) -> array:
        '''
        This function interpolates the texture/color of each vertex according 
        to the given texture and vertUVs coordinate, with the required algo.

        Parameters
        ----------
        texture : array
            The texture image, [WxHx3].
        vertUVs : array
            The UV coordinate of each vertex, [Nx2\.
        algorithm : str
            The algorithm to be used for interpolation. \n
            'nearestNeighbors': faster and less memory demanding, but rough.\n
            'linear': slower and more memory demanding, but more accurate.\n
            'cubic': slowest and most memory demanding, but most accurate.

        Returns
        -------
        array
            The interpolated color of vertices, [Nx3].

        '''
        assert algorithm in ('nearestNeighbors', 'cubic', 'linear'), \
            "only support nearestNeighbors, cubic and linear."
        
        # prepare coodinate
        # Here we assume that the color is at the center of a pixel
        UV_size = texture.shape[0]
        x = np.linspace(1/UV_size/2, 1-1/UV_size/2, UV_size, endpoint=True)
        y = np.linspace(1/UV_size/2, 1-1/UV_size/2, UV_size, endpoint=True)
        meshColor = np.zeros([vertUVs.shape[0], 3])
        
        xg, yg = np.meshgrid(x,y)
        coords = np.stack([xg.flatten(), yg.flatten()], axis = 1)
        
        # interpolate the texture
        if algorithm == 'nearestNeighbors':
            # interp = interpolate.NearestNDInterpolator(coords, texture.reshape(-1, 3))
            meshColor = interpolate.griddata(coords, texture.reshape(-1, 3), (vertUVs[:,0], vertUVs[:,1]), method='nearest')
        elif algorithm == 'linear':
            # interp = interpolate.LinearNDInterpolator(coords, texture.reshape(-1, 3))
            meshColor = interpolate.griddata(coords, texture.reshape(-1, 3), (vertUVs[:,0], vertUVs[:,1]), method='linear')
        else:
            meshColor = interpolate.griddata(coords, texture.reshape(-1, 3), (vertUVs[:,0], vertUVs[:,1]), method='cubic')
        
        return meshColor
            
    
    def colorMeshVertices(self, mesh_in: o3d.geometry.TriangleMesh, 
                          showColor: bool) -> tuple:
        '''
        Given the open3d triangle mesh, this function outputs the textures and
        segmentations of vertices.

        Parameters
        ----------
        mesh_in : o3d.geometry.TriangleMesh
            The mesh to compute the vertex_colors. This mesh must have texture
            and segmentations
        showColor : bool
            Whether to show the interpolated colors and segmentations.

        Returns
        -------
        tuple
            (interpTexture, interpSegment).

        '''
        texture = np.asarray(mesh_in.textures[0])
        segment = np.asarray(mesh_in.textures[1])
        tripUVs = np.asarray(mesh_in.triangle_uvs)
        triplet = np.asarray(mesh_in.triangles).flatten()
        
        _, indices = np.unique(triplet, return_index=True)
        vertTextUV = tripUVs[indices]
        
        interpTexture = self.meshVertexColorInterp(texture, vertTextUV, 'cubic')         
        interpSegment = self.meshVertexColorInterp(segment, vertTextUV, 'nearestNeighbors')   
        
        if showColor:
            
            print('\ndisplaying the interploated texture and segmentation.')
            
            mesh_show_txt = o3d.geometry.TriangleMesh()
            mesh_show_txt.vertices = mesh_in.vertices
            mesh_show_txt.triangles= mesh_in.triangles       
            mesh_show_txt.vertex_colors = o3d.utility.Vector3dVector(interpTexture/255)
            
            mesh_show_clr = o3d.geometry.TriangleMesh()
            mesh_show_clr.vertices = o3d.utility.Vector3dVector( np.asarray(mesh_in.vertices) + np.array([0.6,0,0]) )
            mesh_show_clr.triangles= mesh_in.triangles       
            mesh_show_clr.vertex_colors = o3d.utility.Vector3dVector(interpSegment/255)
            
            o3d.visualization.draw_geometries([mesh_show_txt, mesh_show_clr])
            
        return (interpTexture, interpSegment)


    def meshUpsampleO3D(self, mesh_in: o3d.geometry.TriangleMesh, method: str,
                        numIters: int = 1, visMesh: bool = False, 
                        ) -> o3d.geometry.TriangleMesh:
        '''
        This function uses open3D to upsample the given mesh, i.e. vertices 
        and triangles, and triangle UVs (as open3d does not support this yet).
        
        method: 
            'Simple' = for each triplet, using the mid point of each edge as 
                       vertices to subdivide the tirplet into 4 triplets. The 
                       shape of the mesh would not change.
                       
            'Loop'   = use the algorithm poubished by loop, which will change 
                       the shape of the mesh to make it C2 continuous.
                       
        WARNING: triangle uvs are not handled in open3d subdivide function. It
                 is on the roadmap of open3d team. So, here we implemented 
                 UV upsampling by ourselves.

        Parameters
        ----------
        mesh_in : o3d.geometry.TriangleMesh
            The open3d mesh object to be upsampled.
        method : str
            The algorithm to upsample the mesh.
        numIters : int, optional
            The number of iterations of upsampling; 1 iter = 4 times denser
            The default is 1.
        visMesh : bool, optional
            Wehter to visualize the input and upsampled mesh.
            The default is False.

        Returns
        -------
        o3d.geometry.TriangleMesh
            The upsampled open3d mesh object.

        '''        
        if method == 'simple':
            mesh_dvd = mesh_in.subdivide_midpoint(number_of_iterations=numIters)
        elif method == 'loop':
            mesh_dvd = mesh_in.subdivide_loop(number_of_iterations=numIters)
        
        mesh_dvd = self.meshUVcoordUpsample(mesh_in, mesh_dvd)
                
        if visMesh:
            print('\n\nUpsample algorithm: %s \nmeshin(right) and the upsampled meshout (left)'%method)
            mesh_in.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_in.vertices) + np.array([0.6,0,0]))
            o3d.visualization.draw_geometries([mesh_dvd, mesh_in])
        
        return mesh_dvd
        

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
        txtfile = pjn(path_objectFolder, 'registered_tex.jpg')  # body texture
        smplObj = pjn('/'.join(self.path_SMPLmodel.split('/')[:-1]), 'text_uv_coor_smpl.obj')
    
        # load original SMPL .obj file, to get the triplets and uv coord
        _, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(smplObj)
        smpl_text_uv_coord[:, 1] = 1 - smpl_text_uv_coord[:, 1]    # SMPL .obj need this conversion
        triangle_uvs = smpl_text_uv_coord[smpl_text_uv_mesh.flatten()]
        
        # load the target registered mesh
        dstVertices, Triangle, _, _ = read_Obj(objfile)
        dstTriangulars = Tensor(dstVertices[Triangle.flatten()].reshape([-1, 9])).to(ondevice)
    
        # load SMPL parameters and model; transform vertices and joints.
        registration = pickle.load(open(regfile, 'rb'),  encoding='iso-8859-1')
        SMPLvert_posed, joints = generateSMPLmesh(
            self.path_SMPLmodel, registration['pose'], registration['betas'], 
            registration['trans'], asTensor=True, device = ondevice)
        
        # create open3d body mesh
        bodyMesh_o3d = create_fullO3DMesh(SMPLvert_posed, smplMesh, txtfile, segfile, triangle_uvs, use_text=True)
        
        # upsample the SMPL mesh and triangle_UVs (ours)
        if self.enable_meshdvd:
            bodyMesh_o3d = self.meshUpsampleO3D(bodyMesh_o3d, self.meshdvd_algor, 
                                                self.meshdvd_iters, self.enable_plots)
    
        # interpolate color for all vertices
        colorVertices, segmentations = self.colorMeshVertices(bodyMesh_o3d, self.enable_plots)
    
        # compute the displacements of vetices of SMPL model
        P2PNormDiff = self.computeNormalGuidedDiff( Tensor(array(bodyMesh_o3d.vertices)), dstTriangulars )
        
        # verify the displacements greater than the threshold by their triplets.
        if self.NNfiltering:
            P2PNormDiff = self.meshDisplacementFilter(
                            P2PNormDiff, 
                            Tensor(array(bodyMesh_o3d.triangles)),
                            Tensor(array(bodyMesh_o3d.vertices)), 
                            self.NNfilterIters, self.enable_plots)
        
        # remove displacements on unclothed parts, e.g. face, hands, foots
        if self.onlyCloth:        
            for part, color in _neglectParts_.items():
                mask = (segmentations[:,0] == color[0])* \
                       (segmentations[:,1] == color[1])* \
                       (segmentations[:,2] == color[2])
                P2PNormDiff[mask] = 0
                    
        # save as ground truth displacement and color
        if self.save:
            savePath = pjn(path_objectFolder, 'GroundTruth/')
            if not exists(savePath):
                try:
                    makedirs(savePath)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise        
            print('Saving dispalcements of: ', path_objectFolder)
            with open( pjn(savePath, 'normal_guided_displacements_oversample_%s.npy' \
                           %(['OFF', 'ON'][self.enable_meshdvd]) ), 'wb' ) as f:
                np.save(f, P2PNormDiff)

            print('Saving its textures/colors')
            with open( pjn(savePath, 'vertex_colors_oversample_%s.npy' \
                           %(['OFF', 'ON'][self.enable_meshdvd]) ), 'wb' ) as f:
                np.save(f, colorVertices)
                
            print('Saving its segmentations')
            with open( pjn(savePath, 'segmentations_oversample_%s.npy' \
                           %(['OFF', 'ON'][self.enable_meshdvd]) ), 'wb' ) as f:
                np.save(f, segmentations)
            

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


def verifyOneGroundTruth(path_object: str, path_SMPLmodel: str, 
                          chooseUpsample: bool = True):
    '''
    This function visualize the objects in the given path for verification.

    Parameters
    ----------
    path_object : str
        Path to the folder containing the object.
    path_SMPLmodel : str
        Path to SMPL model.
    chooseUpsample : bool, optional
        Whether to verify the upsampled mesh. The default is True.

    Returns
    -------
    None.

    '''
    # upsampled mesh
    pathDisp = pjn(path_object, 'GroundTruth/normal_guided_displacements_oversample_%s.npy'%
                       (['OFF', 'ON'][chooseUpsample]))
    pathtext = pjn(path_object, 'GroundTruth/vertex_colors_oversample_%s.npy'%
                       (['OFF', 'ON'][chooseUpsample]))
    pathsegm = pjn(path_object, 'GroundTruth/segmentations_oversample_%s.npy'%
                       (['OFF', 'ON'][chooseUpsample]))
    pathobje = pjn('/'.join(path_SMPLmodel.split('/')[:-1]), 'text_uv_coor_smpl.obj')
    
    # SMPL parameters
    pathRegistr  = pjn(path_object, 'registration.pkl')
    
    # load SMPL parameters and model; transform vertices and joints.
    registration = pickle.load(open(pathRegistr, 'rb'),  encoding='iso-8859-1')
    SMPLvert_posed, joints = generateSMPLmesh(
            path_SMPLmodel, registration['pose'], registration['betas'], 
            registration['trans'], asTensor=True)
    
    # load SMPL obj file
    smplVert, smplMesh, smpl_text_uv_coord, smpl_text_uv_mesh = read_Obj(pathobje)
    smpl_text_uv_coord[:, 1] = 1 - smpl_text_uv_coord[:, 1]    # SMPL .obj need this conversion
    
    # upsample the orignal mesh if choose upsampled GT
    if chooseUpsample:
        o3dMesh = o3d.geometry.TriangleMesh()
        o3dMesh.vertices = o3d.utility.Vector3dVector(SMPLvert_posed)
        o3dMesh.triangles= o3d.utility.Vector3iVector(smplMesh) 
        o3dMesh = o3dMesh.subdivide_midpoint(number_of_iterations=1)
        
        SMPLvert_posed = np.asarray(o3dMesh.vertices)
        smplMesh = np.asarray(o3dMesh.triangles)
    
    # create meshes
    if isfile(pathDisp):
        displacement = np.load(pathDisp)
        vertexcolors = np.load(pathtext)
        segmentation = np.load(pathsegm)
        
        growVertices = SMPLvert_posed + displacement
        clrBody = create_fullO3DMesh(growVertices, smplMesh, 
                                     vertexcolor=vertexcolors/vertexcolors.max(), 
                                     use_vertex_color=True)
        
        growVertices += np.array([0.6,0,0])
        segBody = create_fullO3DMesh(growVertices, smplMesh, 
                                     vertexcolor=segmentation/segmentation.max(), 
                                     use_vertex_color=True)    

        o3d.visualization.draw_geometries([clrBody, segBody])
    else:
        print('\n%s does not have %s mesh data.'
              %(path_object.split('/')[-1], ['unsampled', 'upsampled'][chooseUpsample]))
        
        
if __name__ == "__main__":
    
    path_object = '../../datasets/SampleDateset/125611494278283'
    
    with open('../dataset/MGN_render_cfg.yaml') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
        print('\nRendering configurations:\n')
        for key, val in cfgs.items():
            print('%-25s:'%(key), val)   
    
    # mesh_differ = meshDifferentiator(cfgs)
    # mesh_differ.computeDisplacement(path_object)
    
    verifyOneGroundTruth(path_object, cfgs['smplModel_neu'], chooseUpsample=True)
