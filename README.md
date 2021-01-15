# Reconstruction of Dressed Body from a Single RGB Image
The goal of this project is to reconstruct a dressed body and full texture from single RGB images.
To this end, we construct a pipeline to estimate the body pose and shape, per-vertex displacements
(as clothes) and UV texture map explicitly. The implementation is based on [VIBE](https://github.com/mkocabas/VIBE), [GraphCMR](https://github.com/nkolot/GraphCMR), [RotationContinuity](https://github.com/papagina/RotationContinuity), [
TexturePose](https://github.com/geopavlakos/TexturePose), [MultiGarmentNetwork](https://github.com/bharat-b7/MultiGarmentNetwork). 

This repo has been tested on Ubuntu18.04 with GTX1650/GTX2070.


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
- pyopengl
- [mesh](https://github.com/MPI-IS/mesh)

### Thirdparty

- [smpl_webuser](http://smpl.is.tue.mpg.de): please download the package and copy the smpl_webuser folder to .../DEBOR/script/thirdparty. Registration might be required.
- [Multi Garment dataset](https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip): please download and unzip the file to .../DEBOR/datasets/.

## Usage
### Dataset preparation
Scripts for data preparation are in the *../DEBOR/scripts/dataset* folder. The configurations of data preparation is stored in *dataset_preparation_cfg.yaml*. Currently, only the MGN dataset is supported.

If dataset augmentation is not required:
- set "enable_displacement = True" and "enable_rendering=True" in dataset_preparation_cfg.yaml. 
- run MGN_dataPreperation.py

If dataset augmentation is required:
- run MGN_dataPreperation.py with "enable_displacement=True" on and "enable_rendering=False".
- run MGN_dataPreAugmentation.py to augment the dataset. Currently, only pose augmentation is supported.
- run MGN_dataPreperation.py with "enable_displacement=False" and "enable_rendering=True" to render RGB images.

### Training
