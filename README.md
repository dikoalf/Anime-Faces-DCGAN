# Anime-Faces-DCGAN

Baik, sekarang kita sudah memiliki seluruh kode yang diperlukan. Mari kita lanjutkan dan lengkapi README yang telah kita buat sebelumnya:

```markdown
# Anime Faces Generator using DCGAN

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime faces using a dataset from Kaggle.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Libraries and Configuration](#libraries-and-configuration)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-Image
- SciPy
- PSUtil

You can install these libraries using pip:
```bash
pip install tensorflow numpy pandas matplotlib scikit-image scipy psutil
```

## Usage
Clone this repository and navigate to the project directory. Ensure you have the Kaggle dataset downloaded and placed in the correct directory as specified in the code.

## Libraries and Configuration
```python
# Import libraries
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
from scipy.linalg import sqrtm
import psutil
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Dense, Activation, Reshape, BatchNormalization, Conv2DTranspose, LeakyReLU, Flatten,
    Conv2D, Dropout, Input
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Environment Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## Data Collection
The dataset used in this project is sourced from Kaggle. The images are collected and stored in a DataFrame for easy manipulation and shuffling.
```python
# Data Collection
root = '/kaggle/input/anime-faces/data/data'
label = 'Faces'
allFiles = []
for dirname, _, filenames in os.walk(root):
    for filename in filenames:
        allFiles.append(os.path.join(dirname, filename))

imgDF = pd.DataFrame({'filepaths': allFiles, 'label': label})
imgDF = imgDF.sample(frac=1).reset_index(drop=True)
```

## Exploratory Data Analysis (EDA)
A function is defined to display some sample images from the dataset.
```python
# EDA
def showImages(img):
    imgFiles = glob.glob(f'{root}/*')
    fig, ax = plt.subplots(ncols=10, figsize=(30, 3))
    for i in range(10):
        img = plt.imread(imgFiles[i])
        ax[i].imshow(img)
showImages(root)
```

## Model Training
### Image Processing
The images are loaded and preprocessed to be used in training the GAN.
```python
# Image Processing
dataDir = "/kaggle/input/anime-faces/data"
trainData = tf.keras.preprocessing.image_dataset_from_directory(
    dataDir, label_mode=None, color_mode='rgb', batch_size=128, image_size=(128, 128), shuffle=True
).map(lambda x: (x / 127.5) - 1)
trainData = trainData.prefetch(buffer_size=tf.data.AUTOTUNE)
```

### Model Functions
Functions are defined to build the generator and discriminator models.
```python
# Generator Model
def buildGenerator(inputDim):
    init = RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    model.add(Input(shape=(inputDim,)))
    model.add(Dense(8*8*512, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Reshape((8, 8, 512)))
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init))
    model.add(Activation('tanh'))
    return model

# Discriminator Model
def buildDiscriminator(pixel):
    init = RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    model.add(Input(shape=(pixel, pixel, 3)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.2))
    model.add(Conv2D(1, (4, 4), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=init))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model
```

### Training Function
Functions to train the GAN model in chunks and build the GAN model are implemented.
```python
# Train Function in Chunks
def trainModelChunks(gan, generator, discriminator, data, totalEpochs,
                     chunkSize=100, batchSize=16, inputDim=100):
    totalChunks = totalEpochs // chunkSize
    realImagesBatch = next(iter(data.take(1)))
    realImagesBatch = 0.5 * (realImagesBatch + 1)
    overallHistory = {'dLoss': [], 'gLoss': []}

    nextChunkNum = getNextFileNumber('.', 'generator_checkpoint_chunk_', '.weights.h5')

    for chunk in range(totalChunks):
        mem = psutil.virtual_memory()
        if mem.percent >= 60:
            print("RAM usage is high! Stopping training to avoid out of memory error.")
            break

        try:
            print(f"Training chunk {chunk + 1}/{totalChunks}")
            history = trainGan(gan, generator, discriminator, data, epochs=chunkSize, batchSize=batchSize, inputDim=inputDim)
            nextChunkNum += 1
            generator.save_weights(f'generator_checkpoint_chunk_{nextChunkNum}.weights.h5')
            discriminator.save_weights(f'discriminator_checkpoint_chunk_{nextChunkNum}.weights.h5')
            generateAndSaveImg(generator, chunkSize)
            overallHistory['dLoss'].extend(history['dLoss'])
            overallHistory['gLoss'].extend(history['gLoss'])

            realImagesBatch = next(iter(trainData.take(1)))
            realImagesBatch = 0.5 * (realImagesBatch + 1)
            noiseForFid = np.random.normal(0, 1, (realImagesBatch.shape[0], 100))
            generatedImagesForFid = generator.predict(noiseForFid)
            fidScore = calculateFid(model, realImagesBatch, generatedImagesForFid)
            print(f"FID Score : {fidScore}")
        except RuntimeError as e:
            print("Failed to train", e)
            break

        tf.keras.backend.clear_session()

    return overallHistory

# Build Model and Configuration
inputDim = 100
pixel = 64
generator = buildGenerator(inputDim)
discriminator = buildDiscriminator(pixel)
generator.summary()
discriminator.summary()
lastModelCheckPoint = latestCheckpoint(Path().cwd())
if lastModelCheckPoint is not None:
    print(f"Continue from latest checkpoint: {lastModel
