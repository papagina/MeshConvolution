# Mesh Convolution 
This repository contains the implementation (in PyTorch) of the paper [FULLY CONVOLUTIONAL MESH AUTOENCODER USING EFFICIENT SPATIALLY VARYING KERNELS](https://arxiv.org/pdf/2006.04325.pdf) ([Project Page](https://zhouyisjtu.github.io/project_vcmeshcnn/vcmeshcnn.html)). 

## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Citation](#citation)

## Introduction
Here we provide the implementation of convolution,transpose convolution, pooling, unpooling, and residual neural network layers for mesh or graph data with an unchanged topology. We demonstrate the usage by the example of training an auto-encoder for the [D-FAUST dataset](http://dfaust.is.tue.mpg.de/). If you read through this document, it won't be complicated to use our code.

## Usage
### 1. Overview:
The files are organized by three folders: code, data and train. 
**code** contains two programs. *GraphSampling* is used to down and up-sample the input graph and create the *connection matrices* at each step which give the connection between the input graph and output graph. *GraphAE* will load the *connection matrices*to build (transpose)convolution and (un)pooling layers and train an auto-encoder.
**data** contains the template mesh files and the processed feature data.
**Train** stores the *connection matrices* generated by *GraphSampling*, the experiment configuration files and the training results.

### 2. Environment
For compiling and running the C++ project in *GraphSampling*, you need to install cmake, ZLIB and opencv.

For running the python code in GraphAE, I recommend to use anaconda virtual environment with python3.6, numpy, pytorch0.4.1 or higher version such as pytorch1.3, plyfile, json, configparser, tensorboardX, matplotlib, transforms3d and opencv-python.

### 3. Data Preparation
#### Step One: 
Download registrations_f.hdf5 and registrations_m.hdf5 from [D-FAUST](http://dfaust.is.tue.mpg.de/) to data/DFAUST/ and use code/GraphAE/graphAE_datamaker_DFAUST.py to generate numpy arrays, train.npy, eval.npy and test.npy for training, validation and testing, with dimension pc_num*point_num*channel (pc for a model instance, point for vertex, channel for features). For the data we used in the paper, please download from: https://drive.google.com/drive/folders/1r3WiX1xtpEloZtwCFOhbydydEXajjn0M?usp=sharing

#### Step Two: 
Pick up an arbitray mesh in the dataset as the template mesh and create:
1. template.obj. It will be used by *GraphSampling*. If you want to manually assign some center vertices, set their color to be red (1.0, 0, 0) using the paint tool in MeshLab as the example template.obj in data/DFAUST.

2. template.ply. It will be used by *GraphAE* for saving temporate result in ply.

We have put the example templated.obj and template.ply files in data/DFAUST. 

#### Tips:
For any dataset, in general, it works better if scaling the data to have the bounding box between 1.0*1.0*1.0 and 2.0*2.0*2.0.

### 2. GraphSampling
This code will load template.obj, compute the down and up-sampling graphs and write the *connection matrices* for each layer into .npy files. Please refer to Section 3.1, 3.4 and Appendix A.2 in the paper for understanding the algorithms, and read the comments in the code for more details.

For compiling and running the code, go to "code/GraphSampling", open the terminal, run
```
cmake .
make
./GraphSampling
```

It will generate the *Connection matrices* for each sampling layer named as _poolX.npy or _unpoolX.npy and their corresponding obj meshes for visualization in "train/0422_graphAE_dfaust/ConnectionMatrices". In the code, I refer up and down sampling as  "pool" and "unpool" just for simplification. 

*Connection matrix* contains the connection information between the input graph and the output graph. Its dimension is out_point_num*(1+M*2). M is the maximum number of connected vertices in the input graph for all vertices in the output graph. For a vertex i in the output graph, the format of row i is
{N, {id0, dist0}, {id1, dist1}, ..., {idN, distN}, {in_point_num, -1}, ..., {in_point_num, -1}}
N is the number of its connected vertices in the input graph, idX are their index in the input graph, distX are the distance between vertex i's corresponding vertex in the input graph and vertex X (the lenght of the orange path in Figure 1 and 10). {in_point_num, -1} are padded after them. 

For seeing the output graph of layer X, open vis_center_X.obj by MeshLab in vertex and edge rendering mode. For seeing the receptive field, open vis_receptive_X.obj in face rendering mode. 

For customizing the code, open main.cpp and modify the path for the template mesh (line 33) and the output folder (line 46). For creating layers in sequence, use MeshCNN::add_pool_layer(int stride, int pool_radius, int unpool_radius) to add a new down-sampling layer and its corresponding up-sampling layer. When stride=1, the graph size won't change. As an example, in *void set_7k_mesh_layers_dfaust(MeshCNN &meshCNN)*, we create 8 down-sampling and up-sampling layers. 

#### Tips:
The current code doesn't support graph with multiple unconnected components. To enable that, one option is to uncomment line 320 and 321 in meshPooler to create edges between the components based on their euclidean distances.

The distX information is not really used in our network.

### 3. Network Training
#### Step One: Create Configuration files.
Create a configuration file in the training folder. We put three examples 10_conv_pool.config, 20_conv.config and 30_conv_res.config in "train/0422_graphAE_dfaust/". They are the configurations for Experiment 1.3, 1.4 and 1.5 in Table 2 in the paper. I wrote the meaning of each attribute in explanation.config.

By setting the attributes of *connection_layer_lst*, *channel_lst*, *weight_num_lst* and *residual_rate_lst*, you can freely design your own network architecture with all or part of the connection matrices we generated previously. But make sure the sizes of the output and input between two layers match.

#### Step Two: Training
Open graphAE_train.py, modify line 188 to the path of the configuration file, and run
```
python graphAE_train.py
```

It will save the temporal results, the network parameters and the tensorboardX log files in the directories written in the configuration file.

#### Step Three: Testing
Open graphAE_test.py, modify the paths and run
```
python graphAE_test.py
```

#### Tips:
- For path to folders, always add "/" in the end, e.g. "/mnt/.../.../XXX/"

- The network can still work well when the training data are augmented with global rotation and translation.

- In the code, *pcs* means point clouds which refers to all the vertices in a mesh. *weight_num* refers to the size of the kernel basis. *weights* refers to the global kernel basis or the locally-variant kernels for every vertices. *w_weights* refers to the locally variant coefficients for every vertices. 


### 4. Experiments with other graph CNN layers
Check the code in GraphAE27_new_compare and the training configurations in train/0223_GraphAE27_compare
You will need to install the following packages.

pip install torch-scatter==latest+cu92 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu92 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu92 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu92 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric










