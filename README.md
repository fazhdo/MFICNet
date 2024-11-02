# MFICNet

# Highlights

# Environment Setup

# Data Preparation
We utilize two standard datasets (i.e, 7-Scenes and STIVL) to evaluate our method.
* 7-Scenes: The 7-Scenes dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
* STIVL: The STIVL dataset can be downloaded from [STIVL](https://drive.google.com/drive/folders/1dT8gxLmqWeMtdMkLEv1GGUsEI4xmeyDl?usp=sharing).
# STIVL Dataset
STIVL dataset was collected using an Orbbec Astra2 camera with a handhold manner. this dataset records RGB-D images and corresponding camera poses of four different indoor environments. We utilize the dense 3D reconstruction system ROSEFusion to get the pose label. Note that the RGB camera and the depth camera have been transferred to the same coordinate system in advance. 
For each scene, four sequences are recorded, in which three sequences are used for training and one sequence for testing.
![image](https://github.com/fazhdo/STIVL-Dataset/blob/main/%E5%9B%BE%E7%89%871.png)
