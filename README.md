## [1. Autonomous Driving](https://github.com/aamatemanuel/Computer_Vision/tree/main/Autonomous%20Driving)
### Description
This folder contains an overview code of my master's Thesis project at KUL. I'm working along with [Siemens Digital Industries and Software](https://www.sw.siemens.com/en-US/). A general description about my work is related to Trajectory classifications of cars through visual inspection, in two differente domains: 
- _Synthetic domain_. Software [Siemens Prescan](https://plm.sw.siemens.com/en-US/simcenter/autonomous-vehicle-solutions/prescan/).
- _Real domain_. Open source dataset [nuScenes](https://www.nuscenes.org/)

After performing the previously mentioned classification problem, some domain adaptations techniques should be tested to try to generalize a network from synthetic to real domain. Due to confidentiality, I'm just sharing some code to analyze and generate patches of the nuScenes dataset as it is shown in the next picture. Over this kind of patches a classifier is trained.

<p align="center">
<img src="./Autonomous Driving/scene-0048.png" alt="drawing" width="100"/>
</p>  


### Skills
- Deep Learning: Convolutional Neural Networks, LSTM, Temporal Convolutions, Generative Adversarial Networks (PyTorch).
- Classification task for Left and Right turn, through projection of GPS data into pixel domain.
- Pre-processing: Patch generation, data normalization and Data Augmentations (OpenCV).

## [2. Camera Calibration](https://github.com/aamatemanuel/Computer_Vision/tree/main/Camera%20Calibration)
### Description
This is a DIY project, in which I built a low-cost stereo camera. My goal was to learn how the OpenCV methods for camera callibration, undistortion and stereo vision work, so later on I could use this knowledge for different projects. I used as a reference some work done by [LearnOpencv.org](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/) and by the youtuber [Nicola Nielsen](https://www.youtube.com/watch?v=t3LOey68Xpg&ab_channel=NicolaiNielsen-ComputerVision%26AI)

<p align="center">
<img src="./Camera Calibration/Low_cost_stereo_camera.jpeg" alt="drawing" width="350"/>
</p>  

## 3. Computer Vision
During my master studies I took 2 courses relevant to the field:
1. [Computer Vision: Master of Artificial Intelligence.](https://onderwijsaanbod.kuleuven.be/syllabi/e/H02A5AE.htm#activetab=doelstellingen_idm1894640)
2. [Image Analysis and Understanding: Master of Electrical Engineering](https://onderwijsaanbod.kuleuven.be/syllabi/e/H09J2AE.htm#activetab=doelstellingen_idm18554528)  

In those, I had some hands-on experience, I'll describe briefly each of them: 
- [`CV_Traditional_methods`](https://github.com/aamatemanuel/Computer_Vision/blob/main/Computer%20Vision/1_CV_Traditional_methods.py): Understand traditional methods for computer vision, which can be generalized to learning how to use OpenCV. Some of the tested methods were: Color Spaces, Gaussian and Bilateral Filters, thresholding, edged detectors such as sobel operator, hough transform, template matching and morphological transformations.
- [`Machine_Learning`](https://github.com/aamatemanuel/Computer_Vision/blob/main/Computer%20Vision/2_Machine_Learning.ipynb): Build a classifier for face detection through Feature construction such as PCA, HOG and SIFT. Subset of VGG Dataset. Support Vector Machines, Binary classifiers and Convolutional Neural Networks.
- [`Deep_Learning`](https://github.com/aamatemanuel/Computer_Vision/blob/main/Computer%20Vision/3_Deep_Learning.ipynb): Object classification (20 categories) over Pascal VOC dataset, CNN, famous architectures such as Inception modules. Image segmentation U-Net, and Generative Adversarial Networks. 

## [4. Deep Learning](https://github.com/aamatemanuel/Computer_Vision/tree/main/Deep%20Learning)
### Description
For my Thesis project, I started my journey to learn formally Deep Learning technical skills. So I followed multiple sources of information to learn PyTorch (I choose PyTorch over Tensorflow just because in the website [PapersWithCode](https://paperswithcode.com/trends), most of the paper implementations are using PyTorch). The 2 main sources that I followed were [University of Amsterdam](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html), [LearnPyTorch.io](https://www.learnpytorch.io/).  
Additionally, I took some MOOC's for Object Oriented Programming, and pandas for data analysis. 
### Skills
- PyTorch: Tensor, CUDA, Multilayer Perceptron, Torchvision, DL Architectures.
- Object Oriented programming.
- Pandas


## [5. Mobile Robot](https://github.com/aamatemanuel/Computer_Vision/tree/main/Mobile%20Robot)
The [MECO research group @ KU Leuven](https://www.mech.kuleuven.be/en/pma/research/meco) build an AMR for navigation through complex warehouses, with varying dynamic constraints. Now the intention is to add visual perception to the vehicle to detect parking slots and then to automatically perform a parking maneuver. The setup can be shown in the next picture.

<p align="center">
<img src="./Mobile Robot/AMR.png" alt="drawing" width="350"/>
</p>

### Skills
- Camera calibration (specific model is Realsense t265), undistortion, parallel computing
- Line detection workflow: Filtering, Morphological Transformations, Edge Detector, Hough Transform, Bird's Eye View.

The implementation is done through ROS Noetic, and using Docker images. Due to confidentiality, only the code for line detection, and the functions to configure the camera, which were mainly taken from the documentation of the device, are shared. 

