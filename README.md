# DPTN Step Generation

The PyTorch implementation based on DPTN model

- [Paper](https://arxiv.org/abs/2203.02910) (**CVPR2022 Oral**)
- [Github](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network)

![img](https://user-images.githubusercontent.com/26128046/282400375-5260cecc-e9e9-4b59-a105-462b1b2a66d9.png)

Generating complex and fine-grained textures in one step is very challenging, often leading to poor results. 

We propose a step-by-step approach, starting from generating images with coarse-grained textures and progressively moving to fine-grained images.

## Generate Image Step-by-Step

![image](https://user-images.githubusercontent.com/26128046/282400140-283e4a60-3d1a-4a53-a146-0f080b94f01f.png)

### Sampling Coarse-grained Image Process

* discritize constant $d_j∈{1, 2, ⋯255}$: 
  * $d_j>d_(j+1)$
* $j^{th}$ coarse-grained Image $I^j$ can be described 
  * $I^j=Q^j∙d_j$
  * $Q^j$ and $R^j$ represent $j^{th}$the quotient and remainder, respectively.

### Step Prediction (ACGAN)

Building upon the framework of **[Auxiliary Classifier GAN (ACGAN)](https://proceedings.mlr.press/v70/odena17a.html) (PMLR 2017)**, we have incorporated an additional training process: the step prediction loss. This enhancement aims to stabilize the training phase.

#### Key Improvement

- **Step Prediction Loss**: This added component mitigates a common issue in the original ACGAN framework. Without step prediction, the Discriminator often struggles to accurately determine which step of image generation should be executed. This confusion can lead to the generation of artifacts, such as image cracking. By integrating step prediction loss into the training process, we enhance the Discriminator's ability to more accurately guide the generation process, thereby reducing the occurrence of artifacts.

This modification to the ACGAN framework not only stabilizes the training process but also improves the overall quality of the generated images.

![img](https://user-images.githubusercontent.com/26128046/282387633-0e9c8b01-d0b7-4a80-bd1a-35ee52dd9a8d.png)](https://user-images.githubusercontent.com/26128046/282387633-0e9c8b01-d0b7-4a80-bd1a-35ee52dd9a8d.png)

## Installation

#### Requirements

- Python 3
- PyTorch 1.7.1
- CUDA 10.2

#### Conda Installation

```bash
# 1. Create a conda virtual environment.
conda create -n DPTN_step python=3.8
conda activate DPTN_step
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.3

pip install -r requirements.txt
```



## Dataset

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00).

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under the `./dataset/deepfashion` directory.

- We split the train/test set following [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention). Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by runing:

  ```bash
  cd scripts
  ./download_dataset.sh
  ```

  

  Or you can download these files manually：

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the `./dataset/deepfashion` directory.
  - Download the keypoints `pose.rar` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Unzip and put the obtained floder under the `./dataset/deepfashion` directory.

- Run the following code to save images to lmdb dataset.

  ```bash
  python -m scripts.prepare_data \
  --root ./dataset/deepfashion \
  --out ./dataset/deepfashion
  ```

  

## Training

This project supports multi-GPUs training. The following code shows an example for training the model with 256x176 images using 4 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 1234 train.py \
--id $name_of_your_experiment \
--netG dptn \
--batchsize 20 \
 --num_workers 10 \
```



## Inference

- Run the following code to evaluate the trained model:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port 1234 test.py \
  --id $name_of_your_experiment \
  --save_id $save_folder \
  --netG dptn \
  --batchsize 20 \
  --num_workers 10 \
  --simple_test
  ```

  

## Evaluation

```bash
eval_step.py
```



## Results

![img](https://user-images.githubusercontent.com/26128046/282387633-0e9c8b01-d0b7-4a80-bd1a-35ee52dd9a8d.png)![image](https://user-images.githubusercontent.com/26128046/282402331-8da24e55-66b6-4d8a-9115-98a65c8f48f0.png)



Proposed training scheme can generate more detailed texture results than DPTN (baseline).

* Achieved much lower FID score than DPTN

![table](https://user-images.githubusercontent.com/26128046/282402483-ced97f85-c1ea-41e7-aa02-3b7d4e75fe80.png)

## Style Mixing



![image](https://user-images.githubusercontent.com/26128046/282405102-33a5008a-9736-4d16-84b5-f7483038ff0e.png)

![image](https://user-images.githubusercontent.com/26128046/282405135-36324d12-47fe-41fe-b3d1-8a521ecdaec0.png)

![image](https://user-images.githubusercontent.com/26128046/282405062-e927f170-6f19-49de-b686-9c6d571d8c91.png)

We can mix styles without any additional processes, simply by inputting different style images at different steps.

This experiment reveals that in the initial steps, the synthesis primarily involves detailed styles (such as hair, gender, facial features, etc.), and as we progress to the later steps, more universal styles (like clothing color, etc.) are integrated into the synthesis.