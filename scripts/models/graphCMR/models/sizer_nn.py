#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:17:58 2020

currently based on the codes for the paper 
    "SIZER: A DATASET AND MODEL FOR PARSING 3D CLOTHING AND LEARNING SIZE 
     SENSITIVE 3D CLOTHING"

Refer to https://github.com/garvita-tiwari/sizer for details

@Editor: zhantao
"""

import torch.nn as nn
import torch

from .resnet import resnet50


class res50_plus_Dec(nn.Module):
    def __init__(self, latent_size, output_size, dropout=0.3):
        # output_size is 6890*3, currently only support displacements
        # later we can add textures
        super(res50_plus_Dec, self).__init__()
        self.output_size = output_size
        
        enc = [
            resnet50(pretrained=True),
            nn.Linear(2048, latent_size)
            ]
        self.enc = nn.Sequential(*enc)
        
        latent_size2 = latent_size    # add feature dimension here
        
        dec = [
            nn.Linear(latent_size2, int(latent_size2 * 50)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(latent_size2 * 50), int(output_size / 30)),
            nn.ReLU(inplace=True),
            nn.Linear(int(output_size / 30), output_size),
        ]
        self.dec = nn.Sequential(*dec)

    def forward(self, image, pose = None):
        
        if pose is not None:
            print('things to consider:\n'+\
                  '1. should we use axis angle or rotation matrix'+\
                  '2. the problem of continuous representation mentioned in the paper'+\
                  '3. should we append or concatenate to the latent vector or append to all layers with MLP belencing the dimension or both'+\
                  '4. try to support Sizer first')
            raise NotImplementedError('combining pose is not implemented yet, detials see above info')
        
        enc_out = self.enc(image)
        
        # we can append features here
        feat = enc_out
        
        out_dis =  self.dec(feat).view(image.shape[0], int(self.output_size/3), 3)
        return out_dis


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator_size(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_size, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, mesh_A, mesh_B=None):
        # Concatenate image and condition image by channels to produce input
        #not needed fo sizer if we are using only one kind of ibject, we can use betas or something else here or maybe smpl body
        mesh = mesh_A
        if mesh_B is not None:
            mesh = torch.cat((mesh_A, mesh_B),1)
        return self.model(mesh)
