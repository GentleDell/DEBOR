# Reconstruction of Dressed Body from a Single RGB Image
The goal of this project is to reconstruct a dressed body and full texture from single RGB images.
To this end, we construct a pipeline to estimate the body pose and shape, per-vertex offsets
(as clothes) and UV texture map explicitly. The implementation is based on [VIBE](https://github.com/mkocabas/VIBE), [GraphCMR](https://github.com/nkolot/GraphCMR), [RotationContinuity](https://github.com/papagina/RotationContinuity), [
TexturePose](https://github.com/geopavlakos/TexturePose), [MultiGarmentNetwork](https://github.com/bharat-b7/MultiGarmentNetwork). 

This repo has been tested on Ubuntu18.04 with Intel® Core™ i7-9700K CPU and GTX2070 (~19.7FPS).


## Dependencies
### system
- libgl1-mesa-dev (Optional: libglu1, freeglut3)
- boost
### python
- python 3.7
- pytorch
- pytorch3D
- tensorboardx
- scipy
- opencv (>4.0, v3.4.2 does not work)
- matplotlib
- tqdm
- yaml
- chumpy
- trimesh
- addict  (required by open3d)
- plyfile (required by open3d)
- sklearn (required by open3d.ml3d)
- pandas  (required by open3d.ml3d)
- open3d  (>= 0.11.2)
- [opendr](https://github.com/mattloper/opendr)  ( for dataset preparation; install from source with "python setup.py build" and "python setup.py install")
- pyopengl
- [mesh](https://github.com/MPI-IS/mesh)

### Thirdparty
- [smpl_webuser](http://smpl.is.tue.mpg.de): Registration might be required. Please download the package and copy the smpl_webuser folder to .../DEBOR/script/thirdparty (some modifications on pkl load in the codes are needed).
- [neutral SMPL](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl): download and copy the neutral SMPL model to .../DEBOR/body_model/.
- [Multi Garment dataset](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip): download and unzip the datasets to .../DEBOR/datasets/ if you would like to create a new dataset or you can use the provided sample dataset (see [here](### Dataset preparation). 
- [Multi Garment dataset 02](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset_02.zip): download and unzip the datasets to .../DEBOR/datasets/ if you would like to augment the MGN main dataset (the above one).
- [Blender](https://www.blender.org/download/Blender2.91/blender-2.91.2-linux64.tar.xz/): download and unzip the software if you would like to render/generate a dataset.

## Complete structure 
- Under DEBOR/**body_model**:
	- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
	- downConvMat_MGN_sparse.npz
	- edges_smpl.npy
	- mesh_downsampling.npz
	- MGN_train_avgBetas.npy
	- MGN_train_avgPose.npy
	- smpl_mean_params.npz
	- smpl_vt_ft.pkl
	- text_uv_coor_smpl.obj

- Under DEBOR/**datasets**:
	- Multi-Garment_dataset
	- MGN_dataset_02
- Under DEBOR/scripts/**third_party**
    - smpl_webuser

## Usage
### Dataset preparation
Scripts for data preparation are provided in the *DEBOR/scripts/dataset* folder. The configurations of data preparation is stored in *dataset_cfg.yaml*. Currently, only the MGN dataset is supported.

**Steps:**
- set paths and parameters in dataset_cfg.yaml
- set enable_rendering and enable_offsets to True. 
- run MGN_augmentation.py
- run MGN_rendering.py

If augmentation with MGN_dataset_02 is required, set the *number_augment_poses* and *number_augment_garments* to proper value then follow the above steps. 

A sample augmented dataset is provided [here](https://drive.google.com/file/d/18xbbIQSOkmyq3ieNABQeiKK8mNy6vEQo/view?usp=sharing). This dataset generates 4 new subjects for each subjects of the original dataset and renders 48 images from 12 views with 4 lights conditions. But this is not enough to guarantee that the distribution of testing and evaluation datasets are close to that of the training set. (In facts, for offsets, the distributions are different for training, testing and evaluation sets in the sample dataset). So, it is recommended to use the dataset to get familiar with the framework and then create a new dataset having more augmented samples for training, e.g. more augmented poses and garments.

**Example:**

We would like to augment subjects after the 5th; for each subject in MGN main dataset, we would like to have 2 new poses, each having 2 difference garments (i.e. 4 new subjects in total), then we set
```
start_from_ind: 4
number_augment_poses: 2
number_augment_garments: 2
```
The script would compute offsets and augment subjects after the 5th subject. For each src subject there will be 4 new subjects: every 2 of them have the same pose but all of them have different garments. The new pose and garments are randomly sampled.

**Time:**

For number_augment_poses = 2 and number_augment_garments = 2, on Intel® Core™ i7-9700K CPU, it takes around 8 hours to run MGN_augmentation.py on MGN main dataset; For 6 cameras around the body, 4 main lights and 2 horizontal distance, it takes 6 hours to run MGN_rendering.py on the augmented dataset.

### Training
Training related scripts are provided in */DEBOR/scripts/* folder. Training configurations are set in *train_structure_cfg.py*. Train the pipeline with:

```
python train_structure.py 
```
If batch_size=2, a GPU with at least 4GB RAM is required; If batch_size=8, it needs >=8GB GPU RAM; If batch_size=32, it needs >=28GB GPU RAM. Training with large batch_size could lead to smoothened offsets (as clothes).

### Inference/Evaluation
If there is a pre-trained model available, *inference.py* and *evaluation.py* are provided for inference and evaluation. Paths to checkpoints, samples, etc, need to be set **inside** the two scripts. Besides, modify the **config.json** file in the checkpoint folder to make sure all paths in it are correctly set.

Inferencing the model:
```
python inference.py
```

Evaluating the model:
```
python evaluation.py
```
A pre-trained model are provided [here](https://drive.google.com/drive/folders/1VFp0nPBdMYyjKdAWuty7gTUGF7mIWT38?usp=sharing). To use it, download all files and copy them to */DEBOR/logs/local/structure_ver1_full_doubleEnc_newdataset_8_ver1.2/* folder. This model is trained with the above sample dataset.

## References
There are functions, classes and scripts in this project that are borrowed from external repos. Here are some great works we are benefited from: [VIBE](https://github.com/mkocabas/VIBE), [GraphCMR](https://github.com/nkolot/GraphCMR), [RotationContinuity](https://github.com/papagina/RotationContinuity), [
TexturePose](https://github.com/geopavlakos/TexturePose), [MultiGarmentNetwork](https://github.com/bharat-b7/MultiGarmentNetwork). We would  like  to  thank all scientists and researchers in the community for sharing reading and research materials used in this project. 



