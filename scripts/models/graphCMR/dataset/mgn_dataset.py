#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:19:17 2020

@editor: zhantao
"""
from os.path import join as pjn
from glob import glob
from ast import literal_eval

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2

from utils.imutils import crop, flip_img


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    """

    def __init__(self, options, use_augmentation=True, split='train'):
        super(BaseDataset, self).__init__()
        
        # split the dataset into 80%, 10% and 10%
        assert split in ('train', 'test', 'validation')
        self.split = split
        self.options = options
        
        # split objects from train, test and validation
        self.obj_dir = sorted( glob(pjn(options.mgn_dir, '*')) )
        numOBJ = len(self.obj_dir)
        if split == 'train':
            self.obj_dir = self.obj_dir[0:int(numOBJ*0.8)]
        elif split == 'test':
            self.obj_dir = self.obj_dir[int(numOBJ*0.8) : int(numOBJ*0.9)]
        else:
            self.obj_dir = self.obj_dir[int(numOBJ*0.9):]
        
        # load data
        self.imgname, self.center, self.scale = [], [], []
        self.cameraInt, self.cameraExt, self.lights= [], [], []      
        self.GTdispace, self.GTtexture, self.GTdispaceOv, self.GTtextureOv = [], [], [], []
        for obj in self.obj_dir:
            
            # read images and rendering settings
            for path_to_image in sorted( glob( pjn(obj, 'rendering/*smpl_registered.png')) ):
                                
                self.imgname.append( path_to_image )
                cameraPath, lightPath = path_to_image.split('/')[-1].split('_')[:2]
                cameraIdx, lightIdx = int(cameraPath[6:]), int(lightPath[5:])
                
                path_to_rendering = '/'.join(path_to_image.split('/')[:-1])
                with open( pjn( path_to_rendering,'camera%d_boundingbox.txt'%(cameraIdx)) ) as f:
                    # Bounding boxes in mgn are top left, bottom right format
                    # we need to convert it to center and scalse format. 
                    # "center" represents the center in the input image space
                    # "scale"  is the size of bbox compared to a square of 200pix.
                    # In both matplotlib and cv2, image is stored as [numrow, numcol, chn]
                    # i.e. [y, x, c]
                    boundbox = literal_eval(f.readline())
                    self.center.append( [(boundbox[0]+boundbox[2])/2, (boundbox[1]+boundbox[3])/2] )
                    self.scale.append( max((boundbox[2]-boundbox[0])/200, (boundbox[3]-boundbox[1])/200) )
                
                with open( pjn( path_to_rendering,'camera%d_intrinsic_smpl_registered.txt'%(cameraIdx)) ) as f:
                    cameraIntrinsic = literal_eval(f.readline())
                    self.cameraInt.append(cameraIntrinsic)
                
                with open( pjn( path_to_rendering,'camera%d_extrinsic_smpl_registered.txt'%(cameraIdx)) ) as f:
                    cameraExtrinsic = literal_eval(f.readline())
                    self.cameraExt.append(cameraExtrinsic)
                    
                with open( pjn( path_to_rendering,'light%d.txt'%(lightIdx)) ) as f:
                    lightSettings = literal_eval(f.readline())
                    self.lights.append(lightSettings)           
                    
            # read ground truth displacements and texture
            path_to_GT = pjn(obj, 'GroundTruth')
            displace   = np.load( pjn(path_to_GT, 'normal_guided_displacements_oversample_OFF.npy') )
            displaceOv = np.load( pjn(path_to_GT, 'normal_guided_displacements_oversample_ON.npy') )
            texture    = np.load( pjn(path_to_GT, 'vertex_colors_oversample_OFF.npy') )
            textureOv  = np.load( pjn(path_to_GT, 'vertex_colors_oversample_ON.npy') )
            self.GTdispace.append( displace )
            self.GTtexture.append(texture)
            self.GTdispaceOv.append( displaceOv )
            self.GTtextureOv.append(textureOv)
            
        # TODO: change the mean and std to our case
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)       
	
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
	
        self.length = len(self.imgname)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)     # per channel pixel-noise
        rot = 0             # rotation
        sc = 1              # scaling
        if self.split == 'train':
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.options.img_res, self.options.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def __getitem__(self, index):
        item = {}
        center = self.center[index]
        scale = self.scale[index]

        # Get augmentation parameters
        flip,pn,rot,sc = self.augm_params()
        
        # Load image
        imgname = self.imgname[index]
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img_orig'] = img.clone()
        item['img'] = self.normalize_img(img)
        item['imgname'] = imgname

        item['scale'] = float(sc * scale)
        item['center'] = np.array(center).astype(np.float32)
        item['orig_shape'] = orig_shape
        
        item['GTdisplacements'] = self.GTdispace[ index//self.options.img_per_object ]
        item['GTtextures'] = self.GTtexture[ index//self.options.img_per_object ]
        item['GTdisplacements_oversample'] = self.GTdispaceOv[ index//self.options.img_per_object ]
        item['GTtextures_oversample'] = self.GTtextureOv[ index//self.options.img_per_object ]
        
        # Pass path to segmentation mask, if available
        # Cannot load the mask because each mask has different size, so they cannot be stacked in one tensor
        # try:
        #     item['maskname'] = self.maskname[index]
        # except AttributeError:
        #     item['maskname'] = ''
        # try:
        #     item['partname'] = self.partname[index]
        # except AttributeError:
        #     item['partname'] = ''
        return item

    def __len__(self):
        return len(self.imgname)


class MGNDataset(torch.utils.data.Dataset):
    """Mixed dataset with data from all available datasets."""
    
    def __init__(self, options):
        super(MGNDataset, self).__init__()
        self.mgn_dataset = BaseDataset(options)
        # self.mgn_dataset[0]
        self.length = self.mgn_dataset.length
        
        
    def __getitem__(self, i):
        return self.mgn_dataset[i]

    def __len__(self):
        return self.length
    