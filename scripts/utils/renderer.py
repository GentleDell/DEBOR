#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:57:19 2020

@author: zhantao
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        
        blend_params = kwargs.get("blend_params", self.blend_params)
        
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


class renderer(nn.Module):
    def __init__(self, convertToPytorch3D: bool = True, batch_size: int = 1):
        super(renderer, self).__init__()
        
        assert batch_size == 1,\
            'with some issues in pytorch3D, render 1 mesh per forward'
        
        # Setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # prepare covertion matrix
        self.convertToPytorch3D = convertToPytorch3D
        if self.convertToPytorch3D:
            self.GCcdToPytorch3D = torch.Tensor(
                [[[-1, 0, 0], [0,-1,0], [0,0,1]]])\
                .repeat_interleave(batch_size, dim=0).to(self.device)
                        
    def forward(self,
                verts,        # under general camera coordinate rXdYfZ,  N*V*3    
                faces,        # indices in verts to define traingles,    N*F*3
                verts_uvs,    # uv coordinate of corresponding verts,    N*V*2
                faces_uvs,    # indices in verts to define triangles,    N*F*3
                tex_image,    # under GCcd,                            N*H*W*3
                R,            # under GCcd,                              N*3*3 
                T,            # under GCcd,                              N*3
                f,            # in pixel/m,                              N*1
                C,            # camera center,                           N*2
                imgres,       # int
                lightLoc = None):
        
        assert verts.shape[0] == 1,\
            'with some issues in pytorch3D, render 1 mesh per forward'
        
        # only need to convert either R and T or verts, we choose R and T here
        if self.convertToPytorch3D:
            R = torch.matmul(self.GCcdToPytorch3D, R)
            T = torch.matmul(self.GCcdToPytorch3D, T.unsqueeze(-1)).squeeze(-1)
        
        # prepare textures and mesh to render
        tex = TexturesUV(
            verts_uvs = verts_uvs, 
            faces_uvs = faces_uvs, 
            maps = tex_image
        ) 
        mesh = Meshes(verts = verts, faces = faces, textures=tex)
        
        # Initialize a camera. The world coordinate is +Y up, +X left and +Z in. 
        cameras = PerspectiveCameras(
            focal_length=f,
            principal_point=C,
            R=R, 
            T=T, 
            image_size=((imgres,imgres),),
            device=self.device
        )

        # Define the settings for rasterization and shading. 
        raster_settings = RasterizationSettings(
            image_size=imgres, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Create a simple renderer by composing a rasterizer and a shader.
        # The simple textured shader will interpolate the texture uv coordinates 
        # for each pixel, sample from a texture image. This renderer can
        # support lighting easily but we do not iimplement it.
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SimpleShader(
                device=self.device
            )
        )
        
        # render the rendered image(s)
        images = renderer(mesh)
        
        return images
