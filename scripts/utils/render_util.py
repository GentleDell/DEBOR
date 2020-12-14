#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:31:01 2020

@author: zhantao
"""
import os
from glob import glob
from os.path import join as pjn

import numpy as np
from scipy.spatial.transform import Rotation as Rfunc

_EPS_ = 1e-8


class light():
    '''
    This class stores the location and power of a light source.
    '''
    def __init__(self, location: np.array, power: int):
        self.location = location
        self.power = power
    
    def getLocation(self):
        return self.location.tolist()
    
    def getPower(self):
        return self.power
    
    def serialize(self):
        return {"location": self.location.tolist(),
                "power": self.power}


class camera():
    '''
    This class stores parameters to construct a camera. Including intrinsics
    and extrinsics. 
    
    Key attributes:
        The rotation in this class describes the rotation around axes of the 
        'world' coordinate system in the right-hand manner. The location is 
        the position of the camera in the 'world' coordinate system.
            
        The rotation in scipy is in left-hand manner and it support rotation 
        around both the 'world' and 'local' coordinate system.
        
        In camera projection, the rotation has to be around axes of 'local' 
        coordinate system and the location has to be converted accordingly.

        So, since we use scipy as the interface of the conversion in this 
        class, we need to interpret the transformations carefully.        
    
    '''
    def __init__(self, projModel: str, intrinsics: np.array, 
                 extrinsics: np.array = None, resolution: list = [None, None]):
        '''
        This function initiates parameters specifying a camera in 3D space.
        Including intrinsic and extrinsic. 
        
        ATTENTION:
            The **intrinsics** are **not used** in blender yet. But in the 
            member function project(), the intrinsics are used, which should 
            be the intrinsic parameters saved during rendering.
        
        The default rotation mode is "Global", corresponding to the extrinsic
        mode in scipy.transformation.rotation.
        
        Parameters
        ----------
        projModel : str
            The projection model of the camera, 'perspective' or 'orthogonal'.
        intrinsics : np.array
            The intrinsic parameter matrix, shape = 3x3.
        extrinsics : np.array, optional
            The extrinsic parameter matrix [R|t].         
            The default is None.
        
            In blender, both R and t are based on the global/world coordinate 
            system, i.e. t is the coordinate of the camera origin in the world 
            coordinate system and the R (if converted to euler angle) means
            the rotation around the axies of the world coordinate system, 
            instead of camera coordinate system. This is similar to the scipy
            rotation package in extrinsic mode. 
            
            But in scipy, the rotation is left-hand while in most cases and in 
            blender, right-hand rotation is used.
            
            So, this extinsic mat is not the usual extinsic matrix for camera 
            projection. The projection is implemented in projection().
        
        resolution : list, optional
            The resolution of the camera if provided. 
            The default is [None, None].

        Returns
        -------
        None.

        '''
        assert intrinsics.shape == (3,3), 'intrinsic matrix should be 3x3'
        assert projModel in ('perspective', 'orthogonal'), 'only support perspective and orthogonal model'
        self.rotation_format = ('angle_XYZ', 'quaternion', 'matrix')
        
        self.projModel  = projModel
        self.intrinsics = intrinsics
        self.fx = intrinsics[0,0]
        self.fy = intrinsics[1,1]
        self.cx = intrinsics[0,2]
        self.cy = intrinsics[1,2]
        self.resolution_x = [resolution[0], self.cx * 2][resolution[0] is None]
        self.resolution_y = [resolution[1], self.cy * 2][resolution[1] is None]   
        
        if extrinsics is not None:
            assert extrinsics.shape == (3,4)   
        self.extrinsics = extrinsics   # None or 3x4 matrix
        

    def setExtrinsic(self, rotation: np.array, location: np.array, 
                     w2c: str = True):
        '''
        This function sets the extrinsic matrix of the camera from the give 
        rotation and location array.
    
        Parameters
        ----------
        rotation : np.array
            Rotation of the camera, can be euler (XYZ only), quaternion or 
            rotation matrix. The rotation is around axes of the world coord
            system.
        location : np.array
            The location of the camera, shape = 1x3, under the world coodinate
            system.
        w2c : str, optional
            If the rotation and location is to bring world coordinate to the 
            camera coordiante. The default is True.
    
        Returns
        -------
        None.
    
        '''
        if self.extrinsics is not None:
            print("existing camera extrinsics would be overwritten.")
        
        if rotation.size == 3:
            # the rotation in scipy is left-hand (thumb to the positive), but 
            # in our case, we use right-hand rotation, same as blender.
            rotMat = Rfunc.from_euler('XYZ', -rotation.reshape(1,3)).as_matrix()[0]
        if rotation.size == 4:
            # covert from wxyz to xyzw
            raise ValueError("verify the rotaiton of quat in scipy first")
            w = rotation[0]
            rotation[:-1] = rotation[1:]
            rotation[-1]  = w
            rotMat = Rfunc.from_quat(rotation).as_matrix()
        if rotation.size == 9:
            rotMat = rotation
            
        if not w2c:
            raise ValueError('the conversion is not verified.')
            rotMat = rotMat.transpose()
            location = -rotMat@location
            
        self.extrinsics = np.hstack( (rotMat, location[:,None]) )        
        
    def getRotation(self, w2c: bool = True, outformat: str = 'angle_XYZ'): 
        '''
        This function returns the rotation of the camera in the specified 
        direction and in the specified output format.
        
        Parameters
        ----------
        w2c : bool, optional
            The direction of the rotation. If True, from world to the camera.
            Else, from camera to world.
            The default is True.
        outformat : str, optional
            The output format of the rotation. Only 'angle_XYZ', 'quaternion' 
            and 'matrix' are support as output.
            The default is 'angle_XYZ'.

        Returns
        -------
        output : np.array
            The rotation of the camera.

        '''
        assert outformat in self.rotation_format, "unsupported output format %s"%outformat
        assert self.extrinsics is not None, "Extrinsics are not initialized."
        
        
        assert w2c, 'the conversion is not verified.'
        rotMat = [self.extrinsics[:3,:3].transpose(), self.extrinsics[:3,:3]] [w2c]
            
        rotation = Rfunc.from_matrix(rotMat)
        if outformat == "matrix":
            output = rotMat
        elif outformat == "angle_XYZ":
            # Here we should not transpose the rotMat then do the back conversion
            # because the inverse rotation != inverse euler angle.
            output = -rotation.as_euler('XYZ')    # convert it back to angle and then additive inverse
        else:
            raise ValueError("verify the rotaiton of quat in scipy first")
            output = rotation.as_quat()    # x,y,z,w
            # convert to w,x,y,z
            w = output[-1]
            output[1:] = output[0:-1]
            output[0]  = w
            
        return output.tolist()
    
    def getLocation(self, w2c: bool = True): 
        '''
        This function returns the location of the camera in the specified 
        direction.

        Parameters
        ----------
        w2c : bool, optional
            The direction of the location(translation). If True, it brings 
            world coordinate to the camera coordinate.
            The default is True.

        Returns
        -------
        output : TYPE
            Location of the camera, i.e. translation of camera extrinsics.
            
        '''
        assert self.extrinsics is not None, "Extrinsics are not initialized."
        output = self.extrinsics[:,3]
        if not w2c:
            raise ValueError('the conversion is not verified.')
            output = -self.extrinsics[:3,:3].transpose()@output
    
        return output.tolist()
    
    def getResolution(self):
        '''
        This function returns the resolution of the camera.

        Returns
        -------
        Tuple
            Tuple of the resolution of the camera.

        '''
        return (self.resolution_x, self.resolution_y)

    def getGeneralCamera(self):
        '''
        This function returns the intrinsics and extrinsics of the camera in 
        the general cases. I.e. extrinsic is from world to camera; the camera
        coordinate is rightX-downY-frontZ, which is different in blender.

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        # convert the blender camera coordinate system to the one used for 
        # projection, i.e. from rightX-topY-backZ to rightX-downY-frontZ.
        R_blcam2Img = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
        
        # get the rotation and translation
        R = np.array(self.getRotation(outformat='matrix'))
        t = np.array(self.getLocation())
        
        # conver the transformation from blender type to projection type.
        transformGeneral = R_blcam2Img @ np.hstack((R, -R@t[:,None]))
        
        return {'intrinsic': self.intrinsics,
                'extrinsic': transformGeneral}

    def serialize(self):
        return {'rotation': self.getRotation(),
                'location': self.getLocation(),
                'resolution': self.getResolution()}
    
    def projection(self, pointCloud: np.array) -> np.array:
        '''
        This function projects the given point cloud (from .obj of MGN), which 
        is under the world coordinate system, to the image coordinate of the 
        camera. 
        
        The point cloud is from the .obj file of MGN which uses a different 
        coordinate system from the one used in blender.Therefore we convert 
        the coordinate system to the one used in blender first and then do 
        projection to the image plane. 
        
        The coordinate in blender is:
            face toward the -Y; head toward the +Z.
        The coordinate in the .obj file is:
            face toward the +Z; head toward the +Y.
        
        Besides, the camera coordinate system is different from the one used 
        for projection, thus we also further convert the coordinate system 
        before projecting the points to image.

        Parameters
        ----------
        pointCloud : np.array
            Point cloud under the world coordinate system.

        Returns
        -------
        np.array
            The projected pixels on the image plane of the camera.

        '''
        assert self.extrinsics is not None, 'Extrinsics are not initialized.'
        assert pointCloud.shape[0] == 4, 'Shape of the point cloud should be 4xN'
        
        # convert coordinate of points in .obj of MGN to blender coordinate 
        # system, i.e. rotate -90 degree around X in right-hand manner.
        R_vertx2blobj = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0], [0,0,0,1]])
        
        # convert the blender camera coordinate system to the one used for 
        # projection, i.e. from rightX-topY-backZ to rightX-downY-frontZ.
        R_blcam2Img = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
        
        # get the rotation and translation
        R = np.array(self.getRotation(outformat='matrix'))
        t = np.array(self.getLocation())
        
        # conver the transformation from blender type to projection type.
        transformGeneral  = np.hstack((R, -R@t[:,None]))
        
        # transform point cloud to blender camera coordinate
        pointsCameraCoord = transformGeneral@R_vertx2blobj@pointCloud
        
        # convert blender camera coordinate system to projection coord sys and
        # project to the front parallel plane of the camera
        pointsFrontParal  = self.intrinsics@R_blcam2Img@pointsCameraCoord
        pointsFrontParal  = pointsFrontParal[:, pointsFrontParal[-1, :] > 0]
        
        # project points to image coordinate
        pointsImageCoord  = pointsFrontParal/pointsFrontParal[-1,:]
        
        return pointsImageCoord[:2, :]
        

def getMaterialPath(bodyRoot: str):
    '''
    This function stores paths to the materials of the MGN dataset as a list
    of dict.
    
    Members:
    ------
    sampleRootList: list
        the root paths to the objects of MGN.
    meshList: list 
        list of meshes of MGN.
    scanList: list 
        list of scans of MGN.
    meshtextureList: list 
        list of textures of meshes of MGN.
    scantextureList: list 
        list of texture of meshes of MGN.
        
    Returns:
    -------
    List
        A list of dictionary.
    
    '''
    
    assert os.path.isdir(bodyRoot), "The given folder does not exist."
    
    output = []    
    
    for sample in sorted(glob( pjn(bodyRoot, '*') )):
        sample = os.path.abspath(sample)
        sampleDict = {"sampleRootPath": sample,
                      "smpl_registered_path": pjn(sample, 'smpl_registered.obj'),
                      "smpl_registered_texturePath": pjn(sample, 'registered_tex.jpg'),
                      "scan_path": pjn(sample, 'scan.obj'),
                      "scan_texturePath": pjn(sample, 'scan_tex.jpg')}
        output.append(sampleDict)
        
    return output

