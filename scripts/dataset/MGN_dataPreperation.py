#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:31:01 2020

@author: zhantao
"""

import ast
import sys 
sys.path.append( "../" )
sys.path.append( "../third_party/smpl_webuser" )
import pathlib
import argparse
import subprocess
from glob import glob
import pickle as pickle
from os.path import join as pjn

import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils.render_util import light, camera, getMaterialPath
from utils.mesh_util import read_Obj, meshDifferentiator


def _run_blender(blenderPath: str, renderPath: str, cachePath: str, stdin=None,
                 stdout=None, stderr=None):
    '''
    Execute blender in a sub process.
    '''
    return subprocess.run( [blenderPath] + ['--background'] + ['--python'] + [renderPath] + [cachePath],
                           stdin=stdin,
                           stdout=stdout,  # if stdout is not None else subprocess.PIPE,
                           stderr=stderr)

def easyCameras(numCircCameras: int, heights: list, horiz_distance: list, 
                resolution: list) -> list: 
    '''
    This function create a simple list of cameras for rendering.

    Parameters
    ----------
    numCircCameras : int
        The number of cameras in a group, i.e. in a circle.
    heights : list
        A list of height of the cameras.
    horiz_distance : list
        A list of XY distance of cameras to the origin of world coordinate.
    resolution : list
        A list of resolutions of cameras.

    Returns
    -------
    list
        The list of cameras.

    '''
    cameraList = []
    for res in resolution:
        intrinMat = np.array([[1333,  0, res[0]/2], 
                               [0, 1333, res[1]/2],
                               [0,    0,        1]])
        for dist in horiz_distance:
            for h in heights:
                for cnt in range(numCircCameras):
                    
                    rx, ry, rz = np.radians(90), 0.0, np.radians( -360*cnt/numCircCameras )
                    px, py, pz = dist*np.sin(rz), -dist*np.cos(rz), h
                    
                    rotation = np.array([rx, ry, rz])
                    location = np.array([px, py, pz])
                    
                    cam = camera(projModel="perspective", intrinsics=intrinMat)
                    cam.setExtrinsic(rotation, location)
                    cameraList.append(cam.serialize())   
                    
    return cameraList

def easyLights(numLight: int, power: list, initAngle: float = 45, 
               heights: list = [4], horiz_distance: list = [6]) -> list:
    '''
    This function create a simple list of lights for rendering.

    Parameters
    ----------
    numLight : int
        The number of lights in a group, i.e. in a circle.
    power : list
        A list of powers of the lights.
    initAngle : float, optional
        The initial angle of the lights group, in degree. 
        The default is 45.
    heights : list, optional
        A list of height of lights. 
        The default is [4].
    horiz_distance : list, optional
        A list of XY distance of cameras to the origin of world coordinate.
        The default is [6].

    Returns
    -------
    list
        The list of light.

    '''
    lightList = []
    for energy in power:
        for h in heights:
            for dist in horiz_distance:
                for cnt in range(numLight):
                    
                    angle = np.radians(initAngle) - np.radians(360 * cnt/numLight)
                    px, py, pz = dist*np.cos(angle), dist*np.sin(angle), h
                    lightLoc = np.array([px, py, pz])
                    
                    lightSetting = light(location=lightLoc, power = energy)
                    lightList.append(lightSetting.serialize())
                    
    return lightList

def prepareData(cfgs: dict):
    '''
    This function prepares and stores paths, cameras and lights for rendering.

    Parameters
    ----------
    cfgs : dict
        The configuration for rendering.

    Returns
    -------
    None.

    '''
    # get material paths
    MGN_data   = getMaterialPath(cfgs['datarootMGN'])
    
    # create camera list
    cameraList = easyCameras(numCircCameras = cfgs['numCircCameras'], 
                             heights        = cfgs['camera_heights'], 
                             resolution     = cfgs['camera_resolution'],
                             horiz_distance = cfgs['camera_horiz_distance'])
    
    # create light list
    lightList  = easyLights(numLight  = cfgs['numLight'],
                            power     = cfgs['light_power'],
                            heights   = cfgs['light_heights'],
                            initAngle = cfgs['light_initAngle'], 
                            horiz_distance = cfgs['light_horiz_distance'])
    
    # save data to pickle
    with open( pjn(cfgs['cachePath'], 'MGN_data.pickle'), 'wb') as handle:
        pickle.dump(MGN_data, handle)
    with open( pjn(cfgs['cachePath'], 'cameras.pickle'), 'wb') as handle:
        pickle.dump(cameraList, handle)
    with open( pjn(cfgs['cachePath'], 'lights.pickle'), 'wb') as handle:
        pickle.dump(lightList, handle) 
    

def boundingbox(imagefolder: str, cameraIdx: int = 0, marginSize: int = 25):
    '''
    This function projects vertices to the image plane with the saved camera 
    data to verify the rendering quality and then it compute the ground truth 
    bounding box of the body. 

    Parameters
    ----------
    imagefolder : str
        Path to an object of MGN dataset.
    cameraIdx : int, optional
        The index of camera to project the point cloud to. 
        The default is 0.
    marginSize : int, optional
        The size of margin in the bounding box. 
        The default is 25.

    Returns
    -------
    None.

    '''
    # list camera files and obj file
    camIntfile = pjn( imagefolder, 'camera%d_intrinsic_smpl_registered.txt'%(cameraIdx) )
    camExtfile = pjn( imagefolder, 'camera%d_extrinsic_smpl_registered.txt'%(cameraIdx) )
    meshfile   = pjn( '/'.join(imagefolder.split('/')[:-1]), 'smpl_registered.obj' )
    
    # read camera parameters
    with open(camIntfile, "r") as handle:
        cameraIntrinsic = np.array(ast.literal_eval(handle.read()))
    with open(camExtfile, "r") as handle:
        cameraExtrinsic = np.array(ast.literal_eval(handle.read()))    
    
    # read vertices and create camera class
    vertices, _, _, _ = read_Obj(meshfile)
    cameraCLS = camera(projModel="perspective", intrinsics=cameraIntrinsic)
    cameraCLS.setExtrinsic(rotation = cameraExtrinsic[0], 
                           location = cameraExtrinsic[1])
    
    # convert point cloud to homogenous coordinate system and project to image.
    vertices = np.hstack( (vertices, np.ones([vertices.shape[0], 1])) ).transpose()
    pixels = cameraCLS.projection(vertices).astype('int')
    
    # prepare to visualize the projection
    visImage = np.zeros( [int(cameraCLS.resolution_y), int(cameraCLS.resolution_x)] )
    visImage[pixels[1,:], pixels[0,:]] = 1
    plt.imsave( pjn( imagefolder, 'camera%d_light0_projection.png'%(cameraIdx) ), visImage )
    
    # bounding box of the body
    max_col = max(min(pixels[0,:].max() + marginSize, cameraCLS.resolution_x), 0)
    min_col = min(max(pixels[0,:].min() - marginSize, 0), cameraCLS.resolution_x) 
    max_row = max(min(pixels[1,:].max() + marginSize, cameraCLS.resolution_y), 0)
    min_row = min(max(pixels[1,:].min() - marginSize, 0), cameraCLS.resolution_y)
    
    with open(pjn( imagefolder, 'camera%d'%cameraIdx+'_boundingbox.txt'), "w") as output:
        output.write(str([min_row, min_col, max_row, max_col]))
        

if __name__ == "__main__":
    '''
    Renderning images for meshes of the MGN dataset and computing the ground 
    truth displacements, colors, and segmentations for vertices.
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_preparation_cfg', action='store', type=str, 
                        help='Path to the configuration for rendering.', 
                        default = './dataset_preparation_cfg.yaml')
    args = parser.parse_args()

    # read preparation configurations
    with open(args.dataset_preparation_cfg) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
        print('\nDataset preparation configurations:\n')
        for key, val in cfgs.items():
            print('%-25s:'%(key), val)    
    
    # require confirmation
    if cfgs['requireConfirm']:
        msg = 'Do you confirm that the settings are correct?'
        assert input("%s (Y/N) " % msg).lower() == 'y', 'Settings are not comfirmed.'
    
    if cfgs['enable_augmentation']:
        outstr = \
        ''' 
        To do augmentation, folders of subjects must have the "/GroundTruth"
        subfolder containing displacements for augmentation.
    
        Thus, a proper order of the dataset properation is:
        1. run MGN_dataPreperation.py with 'enable_displacement' on and 
        "enable_rendering" off.
        
        2. run MGN_dataPreAugmentation.py to augment the dataset.
        
        3. run MGN_dataPreperation.py with 'enable_displacement' off and 
        "enable_rendering" on to render RGB images.
        
        No operation would be conducted after showing this message.
        '''
        print(outstr)
        sys.exit(0)
    
    # prepare the data for rendering
    prepareData(cfgs)
    
    # set paths and use blender to render images
    if cfgs['enable_rendering']:
        renderScript = pjn( str(pathlib.Path().absolute().parent), "utils/blender_util.py" )        
        cachePathAbs = str(pathlib.Path(__file__).parent.absolute())
        _run_blender(cfgs['blenderPath'], renderPath=renderScript, cachePath = cachePathAbs)
    
    # compute the ground truth displacements; verify the quality of the 
    # rendered image and get the bounding box
    mesh_differ = meshDifferentiator(cfgs)
    for folder in sorted(glob( pjn(cfgs['datarootMGN'], '*' ) )):
        
        # GT displacements
        if cfgs['enable_displacement']:
            print("computing the GT displacements for: ", folder)
            mesh_differ.computeDisplacement(folder)
        
        # rendering verification and generate boundingbox
        if cfgs['enable_rendering']:
            print("verifying camera parmaeters.")
            numCameras = cfgs['numCircCameras'] * len(cfgs['camera_heights']) * \
                          len(cfgs['camera_horiz_distance']) * len(cfgs['camera_resolution'])
            for cameraIdx in range(numCameras):
                boundingbox( pjn(folder, 'rendering'), cameraIdx, marginSize = 10)
    
    
    
    