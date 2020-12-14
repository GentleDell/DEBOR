#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:19:17 2020

@editor: zhantao
"""
from os.path import join as pjn, isdir
from glob import glob
from ast import literal_eval
import pickle as pickle

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2

from utils.render_util import camera as cameraClass
from utils.imutils import crop, flip_img, background_replacing


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    """

    def __init__(self, options, use_augmentation=True, split='train'):
        super(BaseDataset, self).__init__()
        
        # store options
        self.options = options
        
        # split the dataset into 80%, 10% and 10% 
        # for train, test and validation
        assert split in ('train', 'test', 'validation')
        self.split = split
        
        self.obj_dir = sorted( glob(pjn(options.mgn_dir, '*')) )
        numOBJ = len(self.obj_dir)
        if split == 'train':
            self.obj_dir = self.obj_dir[0:int(numOBJ*0.8)]
        elif split == 'test':
            self.obj_dir = self.obj_dir[int(numOBJ*0.8) : int(numOBJ*0.9)]
        else:
            self.obj_dir = self.obj_dir[int(numOBJ*0.9):]
        
        # background image dir 
        #    for rendered images, their background will be replaced by random 
        #    images in the folder, if required.
        #
        #    training images and testing/validation image have backgrounds 
        #    from differernt folder.
        if options.replace_background:
            self.bgimages = []
            imgsrc = ('images/validation/*', 'images/training/*')[split == 'train']
            for subfolder in sorted(glob(pjn(options.bgimg_dir, imgsrc))):
                for subsubfolder in sorted( glob( pjn(subfolder, '*') ) ):
                    if 'room' in subsubfolder:
                        self.bgimages += sorted(glob( pjn(subsubfolder, '*.jpg'))) 
            assert len(self.bgimages) > 0, 'No background image found.'
        else:
            self.bgimages = None
        
        # load datatest
        self. objname, self.imgname, self.center, self.scale = [], [], [], []
        self.camera, self.lights= [], []      
        self.meshGT, self.smplGTParas, self.UVmapGT = [], [], []
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
                
                # read and covert camera parameters
                with open( pjn( path_to_rendering,'camera%d_intrinsic_smpl_registered.txt'%(cameraIdx)) ) as f:
                    cameraIntrinsic = literal_eval(f.readline())     
                with open( pjn( path_to_rendering,'camera%d_extrinsic_smpl_registered.txt'%(cameraIdx)) ) as f:
                    cameraExtrinsic = literal_eval(f.readline())                    
                camthis = cameraClass(
                    projModel  ='perspective',
                    intrinsics = np.array(cameraIntrinsic))
                camthis.setExtrinsic(
                    rotation = np.array(cameraExtrinsic[0]), 
                    location = np.array(cameraExtrinsic[1]))
                self.camera.append( camthis.getGeneralCamera() )
            
                # light settings
                with open( pjn( path_to_rendering,'light%d.txt'%(lightIdx)) ) as f:
                    lightSettings = literal_eval(f.readline())
                    self.lights.append(lightSettings)           
                    
            # read ground truth displacements and texture vector
            path_to_GT = pjn(obj, 'GroundTruth')
            displace   = np.load( pjn(path_to_GT, 'normal_guided_displacements_oversample_OFF.npy') )
            # displaceOv = np.load( pjn(path_to_GT, 'normal_guided_displacements_oversample_ON.npy') )
            texture    = np.load( pjn(path_to_GT, 'vertex_colors_oversample_OFF.npy') )
            # textureOv  = np.load( pjn(path_to_GT, 'vertex_colors_oversample_ON.npy') )
            self.meshGT.append({'displacement': displace,
                                'texture': texture})
            
            # read smpl parameters
            registration = pickle.load(open( pjn(obj, 'registration.pkl'), 'rb'),  encoding='iso-8859-1')
            self.smplGTParas.append({'betas': registration['betas'],
                                     'pose':  registration['pose'], 
                                     'trans': registration['trans']})
            
            # read and resize UV texture map same for the segmentation 
            UV_textureMap = cv2.imread( pjn(obj, 'registered_tex.jpg') )[:,:,::-1]/255.0
            UV_textureMap = cv2.resize(UV_textureMap, (self.options.img_res, self.options.img_res), cv2.INTER_CUBIC)
            if "_pose" in obj.split('/')[-1]:
                UV_textureMap = np.flip(UV_textureMap, axis = 0)
            self.UVmapGT.append(UV_textureMap)
            
            self.objname.append(obj.split('/')[-1])
            
        # TODO: change the mean and std to our case
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)       
	
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation and self.options.use_augmentation
        self.use_augmentation_rgb = use_augmentation and self.options.use_augmentation_rgb
        
        # lenght of dataset
        self.length = len(self.imgname)
        
        # define the correspondence between image and background in advance
        if options.replace_background:
            self.bgperm = np.random.randint(0, len(self.bgimages), size = self.length)

        # img = cv2.imread(self.imgname[251])[:,:,::-1].copy().astype(np.float32)
        # bgimg = cv2.imread(self.bgimages[251])[:,:,::-1].copy().astype(np.float32)
        # img = background_replacing(img, bgimg)
        # plt.imshow(img/255)

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
        # crop and rotate the image
        if self.use_augmentation:
            rgb_img = crop(rgb_img, center, scale, 
                          [self.options.img_res, self.options.img_res], rot=rot)
        else:
            rgb_img = crop(rgb_img, center, scale, 
                          [self.options.img_res, self.options.img_res], rot=0)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        if self.use_augmentation_rgb:
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
        
        # read background image and replace the background if available
        if self.bgimages is not None:
            bgimgname = self.bgimages[ self.bgperm[index] ]
            try:
                bgimg = cv2.imread(bgimgname)[:,:,::-1].copy().astype(np.float32)
            except TypeError:
                print(bgimgname)
            img = background_replacing(img, bgimg)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        
        # Store image before normalization to use it in visualization
        item['img_orig'] = img.clone()
        item['img'] = self.normalize_img(img)
        item['imgname'] = imgname
        item['objname'] = self.objname[ index//self.options.img_per_object ]
 
        item['scale'] = float(sc * scale)
        item['center'] = np.array(center).astype(np.float32)
        item['orig_shape'] = orig_shape
        item['camera'] = self.camera[index]
        
        item['meshGT'] = self.meshGT[ index//self.options.img_per_object ]
        item['smplGT'] = self.smplGTParas[ index//self.options.img_per_object ]
        item['UVmapGT']= self.UVmapGT[ index//self.options.img_per_object ]    # no need to touch the GT
        
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
    
    def __init__(self, options, split='train'):
        super(MGNDataset, self).__init__()
        self.mgn_dataset = BaseDataset(options, 
                                       use_augmentation=True, 
                                       split=split)
        # self.mgn_dataset[0]
        self.length = self.mgn_dataset.length
        
        
    def __getitem__(self, i):
        return self.mgn_dataset[i]

    def __len__(self):
        return self.length
    