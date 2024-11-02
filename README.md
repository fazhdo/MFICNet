# MFICNet

MFICNet: A Multi-modality Fusion Network with Information Compensation for Accurate Indoor Visual Localization
# Highlights

As a crucial technology in numerous visual applications, visual localization has been extensively studied, with an effective solution known as scene coordinate regression (SCoRe). Generally, SCoRe methods generate scene coordinates using convolutional neural networks (CNNs) and then determine the camera pose with a PnP algorithm. While these methods demonstrate impressive localization accuracy, they primarily rely on a single modality, e.g., RGB camera, which leads to texture dependency and structural ambiguity problems. Specifically, perceptual confusion caused by similar image textures in real indoor scenes causes a severe decline in localization accuracy, as the performance of the networks heavily depends on the semantic information of objects. Additionally, current methods struggle to robustly recover structural details of objects because RGB images lack 3D geometric structural information. We believe that these two issues stem from the inherent limitations of single-modality. There is potential for complementarity between semantic and structural information. Towards this end, we propose MFICNet, a novel visual localization network that investigates the feasibility of simultaneously utilizing RGB and depth images to achieve accurate visual localization. Technically, MFICNet employs a heterogeneous backbone to extract features from RGB images and depth images separately. The structural feature obtained from depth images enhances the identifiability of similar image patches and imposes structural constraints for scene coordinates. After that, an information compensation module is introduced to evaluate the contributions of semantic and structural features and perform deep fusion to generate discriminative features.

# Environment Setup

* Create environment:
  
```python
conda create -n MFICNet python=3.7
conda activate MFICNet
```

* Install torch:
  
```python
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

* Dependencies:
  
MFICNet is built based on [openmmlab](https://github.com/open-mmlab).
```python
mmcv:
pip install mmcv==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```
Download source code from: [mmdetection-2.24.1](https://github.com/open-mmlab/mmdetection), [mmdetection3d-1.0.0rc4](https://github.com/open-mmlab/mmdetection3d), and [mmsegmentation-0.24.0](https://github.com/open-mmlab/mmsegmentation), [mmclassification-v0.23.1](https://github.com/open-mmlab?q=&type=all&language=&sort=).
```python
mmdetection-2.24.1:
unzip mmdetection-2.24.1.zip
cd mmdetection-2.24.1
pip install .

mmdetection3d-1.0.0rc4:
unzip mmdetection3d-1.0.0rc4.zip
cd mmdetection3d-1.0.0rc4
pip install .

mmsegmentation-0.24.0:
unzip mmsegmentation-0.24.0.zip
cd mmsegmentation-0.24.0
pip install .

mmclassification-v0.23.1:
unzip mmclassification-v0.23.1.zip
cd mmclassification-v0.23.1
pip install .
```

Enter the file path: mmclassification-v0.23.1\mmcls, copy the code file to the corresponding location.

* Training
  
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 bash /tools/dist_train.sh ./configs/fdanet/SERVER.py 4 --work-dir path/to/save/weights/
```
* Inference
  
```python
CUDA_VISIBLE_DEVICES=0 python /tools/test.py ./configs/fdanet/XXX.py /path/to/checkpoints/ --metrics accuracy
```

# Model weights
We released the weights trained on the 7scenes dataset [here](https://drive.google.com/drive/folders/1dT8gxLmqWeMtdMkLEv1GGUsEI4xmeyDl?usp=sharing).

# Data Preparation
We utilize two standard datasets (i.e, 7-Scenes and STIVL) to evaluate our method.
* 7-Scenes: The 7-Scenes dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
* STIVL: The STIVL dataset can be downloaded from [STIVL](https://drive.google.com/drive/folders/1dT8gxLmqWeMtdMkLEv1GGUsEI4xmeyDl?usp=sharing).
# STIVL Dataset
STIVL dataset was collected using an Orbbec Astra2 camera with a handhold manner. this dataset records RGB-D images and corresponding camera poses of four different indoor environments. We utilize the dense 3D reconstruction system ROSEFusion to get the pose label. Note that the RGB camera and the depth camera have been transferred to the same coordinate system in advance. 
For each scene, four sequences are recorded, in which three sequences are used for training and one sequence for testing.
![image](https://github.com/fazhdo/STIVL-Dataset/blob/main/%E5%9B%BE%E7%89%871.png)
