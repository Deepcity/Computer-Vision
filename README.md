All Chinese note release on https://blog.keboe.cn

# Compute Vision

This repository is restoring CV-lreanning recordings by target and stages.

**Now using learning structure**

1. pyTorch
2. MindSpore

**planning list:** 

1. Supervised Learning
   1. Classifier
      1. clothes Classifier
      2. handwriting recognition
      3. Distinguish between cat and dog (Binary classification)
2. Unsupervied Learning
   1. Clustering Model
3. (waiting for update)

**Implemention:**

1. CLothes Sorter: this Classifier can class 10 kinds of clothes, builded according [microsoft-lreanning](https://learn.microsoft.com/zh-cn/training/modules/intro-machine-learning-pytorch/3-data).
   - status: closed
2. handwriting recognition
   - status：Implementing
3. Distinguish between cat and dog
   - status：Implementing
4. (waiting for update)

## Clothes Sorter

Directory Interpretation:

1. main.py: Early version code document, recording all python codes.
2. data: storage training data and test data, classifiy by name tag.
3. paramters-config: model paramters output directory.
4. notebook: storage some note form kinds of sources.

Model:

input layer : $1\times 28\times 28 \rightarrow 512$

hidden layer: $512\rightarrow 512$ 

output layer: $512 \rightarrow 10$

## MNIST

