"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

from .graph_layers import GraphResBlock, GraphLinear

class GraphCNN(nn.Module):
    
    def __init__(self, A, ref_vertices, infeature=2048, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.infeature = infeature
        
        layers = [GraphLinear(3 + infeature, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)

    def forward(self, image_enc, pose = None):
        """Forward pass
        Inputs:
            x: size = (B, self.infeature)
            pose : size = (B, 1, 24)
        Returns:
            Regressed non-parametric displacements: size = (B, 6890, 3)
        """
        
        if pose is not None:
            print('things to consider:\n'+\
                  '1. should we use axis angle or rotation matrix'+\
                  '2. the problem of continuous representation mentioned in the paper'+\
                  '3. should we append or concatenate to the latent vector or append to all layers with MLP belencing the dimension or both'+\
                  '4. try to support Sizer first')
            raise NotImplementedError('combining pose is not implemented yet, detials see above info')
        
        batch_size = image_enc.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)

        x = image_enc.view(batch_size, self.infeature, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, x], dim=1)
        x = self.gc(x)
        shape = self.shape(x).permute(0,2,1)
        
        return shape
