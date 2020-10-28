#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:36:43 2020

@author: zhantao
"""

import sys
import errno
from glob import glob
from os import makedirs
from os.path import exists, join as pjn 
import pickle as pickle

import bpy

def blenderCamera(camera, scene):
    '''
    This function computes the camera in the blender scene.

    Parameters
    ----------
    camera : bpy.data.objects["Camera"].data
        Camera data object of blender.
    scene : bpy.context.scene
        scene object of blender.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    '''
    # assume image is not scaled
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert camera.sensor_fit != 'VERTICAL'
    
    f_in_mm = camera.lens
    sensor_width_in_mm = camera.sensor_width
    
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    
    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    
    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect
    
    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - camera.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * camera.shift_y
    
    K = [[f_x, 0, c_x],
         [0, f_y, c_y],
         [0,   0,   1]]
        
    return K


def render_MGN(camList: list, lightList: list, fileList: list, 
               meshName: str = 'smpl_registered'):  
    '''
    This function use blender to render the meshes of MGN dataset with the 
    given cameras and lightings. Tested on Blender2.90.1.

    Parameters
    ----------
    camList : list
        A list of camera, defines the intrinsics and extrinsics of cameras.
        ATTENTION:
            The definition of extrinsics is: p' = R(p - t). This is different
            from some of other packages and libs.
    lightList : list
        A list of light, defines the power and location of cameras..
    fileList : list
        A list of dict of paths to the materials of the MGN dataset.
    meshName : str, optional
        The mesh to render. MGN provides "smpl_registered" and "scan".
        The default is 'smpl_registered'.

    Returns
    -------
    None.

    '''
    assert meshName in ('smpl_registered', 'scan'), "MGN only provides smpl_registered and scan"
    
    # align the name
    bpy.data.objects['Cube'].name = meshName
    
    # define pointer to scene
    scene = bpy.context.scene
    
    # set render engine and device
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    
    # enable nodes    
    scene.use_nodes = True
    tree = scene.node_tree
    
    # create image processing node
    alpha_node = tree.nodes.new(type='CompositorNodeAlphaOver')
    alpha_node.inputs[1].default_value = (0,255,0,1)
    alpha_node.premul = 1.0
    
    # link nodes
    tree.links.new(tree.nodes['Composite'].inputs['Image'], tree.nodes['Alpha Over'].outputs['Image'])
    tree.links.new(tree.nodes['Alpha Over'].inputs[2], tree.nodes['Render Layers'].outputs['Image'])

    # for each object
    for mats in fileList:
        
        # create the folder to save the rendered images and settings
        renderPath = pjn(mats['sampleRootPath'], 'rendering/')
        if not exists(renderPath):
            try:
                makedirs(renderPath)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        # remove existing objects and texture
        bpy.data.objects[2].select_set(True)
        bpy.ops.object.delete() 
        bpy.data.materials.remove(bpy.data.materials[0])
        
        # load new object and disselect it 
        bpy.ops.import_scene.obj( filepath = mats[ meshName+'_path' ] )
        bpy.data.objects[2].select_set(False)
        
        # create mesh texture node for rendering if no texture exist
        texImageNode = bpy.data.materials[0].node_tree.nodes.new('ShaderNodeTexImage')
        bpy.data.materials[0].node_tree.links.new( bpy.data.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'], texImageNode.outputs['Color'])
        texImageNode.image = bpy.data.images.load(mats[ meshName+'_texturePath'])

        # for each camera 
        for camIdx, camera in enumerate(camList):
            # remove existing camera
            bpy.data.objects[0].select_set(True) 
            bpy.ops.object.delete()
            
            # create a new camera and link it to the scene
            camera_data = bpy.data.cameras.new(name='Camera')
            camera_object = bpy.data.objects.new('Camera', camera_data)
            scene.collection.objects.link(camera_object)

            # set the parameters of the camera and renderer
            bpy.data.objects[0].location = camera['location']
            bpy.data.objects[0].rotation_euler = camera['rotation']
            scene.render.resolution_x = camera['resolution'][0]
            scene.render.resolution_y = camera['resolution'][1]
            
            # verify that the camera matrix in blender is correct 
            camerabl  = bpy.data.objects[0].data
            cameraMat = blenderCamera(camerabl, scene)

            # save the camera intrinsic parameters to the root folder
            # Attention:
            #    in Blender, the default sensor size is 32*24mm but it is not necessarily fully
            #    use. I.e. it could be 32*18 if the size is 16:9. Pixel shape is square.
            with open(pjn( mats['sampleRootPath'], 'rendering/camera%d'%camIdx+'_intrinsic_'+meshName+'.txt'), "w") as output:
                output.write(str(cameraMat))
                
            # save the camera extrinsic parameters to the root folder
            # Attention:
            #    the rotation is the euler angle around the XYZ axes of the 'world'('global') 
            #    coordinate system, instead of the camera coordinate system, in X, Y, Z order.
            #    the location is under the 'world'('global') coordinate system.
            with open(pjn( mats['sampleRootPath'], 'rendering/camera%d'%camIdx+'_extrinsic_'+meshName+'.txt'), "w") as output:
                output.write( str([camera['rotation'], camera['location']]) )
                
                
            # for each light settings
            for lightIdx, lightSetting in enumerate(lightList):

                # remove existing light
                bpy.data.objects[1].select_set(True) 
                bpy.ops.object.delete() 

                # create a new light at given position with given power
                bpy.ops.object.light_add( location = lightSetting['location'] )
                bpy.data.objects[1].name = 'Light'
                light = bpy.data.objects['Light']
                light.data.energy = lightSetting['power']
                light.select_set(False)

                # select the camera for rendering
                scene.camera = bpy.data.objects[0]
                scene.render.filepath = pjn( mats['sampleRootPath'], 'rendering/camera%d'%camIdx+'_'+'light%d'%lightIdx+'_'+meshName+'.png')
                bpy.ops.render.render(write_still = True)
                
                # save the light position
                #    the location is under the 'world'('global') coordinate system.
                with open(pjn( mats['sampleRootPath'], 'rendering/light%d'%lightIdx+'.txt'), "w") as output:
                    output.write( str([camera['rotation'], camera['location']]) )
                

if __name__ == "__main__":
    '''
    Execute blender to render images.
    '''
    
    assert len(sys.argv) == 5, "The path to the folder containing the pickles should be given."
    
    pickleRoot = sys.argv[-1]
    with open( pjn( pickleRoot, 'MGN_data.pickle' ), 'rb') as handle:
        MGN_data = pickle.load(handle)
    with open( pjn( pickleRoot, 'cameras.pickle'), 'rb') as handle:
        cameras  = pickle.load(handle)
    with open( pjn( pickleRoot, 'lights.pickle'), 'rb' ) as handle:
        lights   = pickle.load(handle)

    render_MGN(cameras, lights, MGN_data)
        
        