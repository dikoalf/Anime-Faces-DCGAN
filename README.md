Seluruh bagian README yang telah kamu susun tampak sangat baik dan komprehensif. Berikut adalah README lengkap untuk proyekmu:

# Anime Faces Generator using DCGAN

Welcome to the Anime Faces Generator repository! This project is a part of my exploration into Generative Adversarial Networks (GAN) architecture, specifically using Deep Convolutional GANs (DCGAN). By leveraging a dataset from Kaggle, I aim to generate high-quality anime-style faces, pushing the boundaries of what GANs can achieve.

The core objective of this project is to deepen my understanding of GAN architectures and their potential applications in creating realistic synthetic images. By following the instructions in this repository, you can replicate my training process, generate your own anime faces, and evaluate the model's performance.

This exploration not only serves as a stepping stone in my learning journey but also contributes to the broader field of generative models. Dive in and join me in uncovering the fascinating world of AI-driven art creation!

Kaggle Dataset : https://www.kaggle.com/datasets/soumikrakshit/anime-faces

## Table of Contents
- [Installation](##installation)
- [Usage](##usage)
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
In this section, we define the functions to build the generator and discriminator models for our DCGAN. The architecture is inspired by the 2016 paper titled "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz, and Soumith Chintala. According to the paper, stable Deep Convolutional GANs (DCGANs) should adhere to the following architectural guidelines:
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batch normalization (batchnorm) in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in the generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

#### Generator Model
The generator model is responsible for generating new images from random noise. It uses several Conv2DTranspose layers to upsample the input noise and produce a final image output.
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
```
#### Discriminator Model
The discriminator model is designed to distinguish between real images from the dataset and fake images generated by the generator. It uses several Conv2D layers with strided convolutions and LeakyReLU activations.
```python
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
The training functions for the GAN model are implemented to handle the training process in a modular and robust way. This section includes two main functions: trainGan and trainModelChunks.

#### trainGan Function
The trainGan function is responsible for training the GAN model for a specified number of epochs. It updates both the generator and discriminator models, and maintains a history of their losses.

1. Initialize Variables: Set up the batch size, and initialize dictionaries to store the discriminator and generator loss history.
2. Epoch Loop: Loop through the specified number of epochs.
3. Batch Processing: For each batch in the training data:
    - Resize and Normalize Real Images: Resize images to 64x64 and normalize them.
    - Generate Fake Images: Generate fake images using random noise as input to the generator.
    - Label Smoothing and Noise Injection: Apply label smoothing and noise injection to the labels.
    - Train Discriminator: Update the discriminator with both real and fake images.
    - Train Generator: Update the generator with the goal of fooling the discriminator into classifying fake images as real.
4. Log Losses: Calculate and log the average losses for each epoch.
5. Print Progress: Print the progress of training for each epoch.
```python
def trainGan(gan, generator, discriminator, trainData, epochs, batchSize, inputDim):
    halfBatch = batchSize // 2
    history = {'dLoss': [], 'gLoss': []}

    for epoch in range(epochs):
        epochDLoss = []
        epochGLoss = []

        for realImages in trainData:
            # Resize and normalize real images
            realImages = tf.image.resize(realImages, [64, 64], method=tf.image.ResizeMethod.BICUBIC)
            realImages = (realImages - 0.5) * 2

            # Generate fake images
            noise = tf.random.normal((halfBatch, inputDim))
            fakeImages = generator(noise, training=True)

            # Label smoothing and noise injection
            realLabels = tf.random.uniform((halfBatch, 1), 0.9, 1.0)
            fakeLabels = tf.zeros((halfBatch, 1))

            # Train Discriminator
            discriminator.trainable = True
            dLossReal = discriminator.train_on_batch(realImages[:halfBatch], realLabels)
            dLossFake = discriminator.train_on_batch(fakeImages, fakeLabels)
            dLoss = 0.5 * np.add(dLossReal, dLossFake)
            epochDLoss.append(dLoss)

            # Train Generator
            discriminator.trainable = False
            noise = tf.random.normal((batchSize, inputDim))
            gLoss = gan.train_on_batch(noise, tf.ones((batchSize, 1)))
            epochGLoss.append(gLoss)

        # Average loss per epoch
        avgDLoss = np.mean(epochDLoss)
        avgGLoss = np.mean(epochGLoss)
        history['dLoss'].append(avgDLoss)
        history['gLoss'].append(avgGLoss)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | D loss: {avgDLoss:.4f}, G loss: {avgGLoss:.4f}")

    return history
```
#### trainModelChunks Function
The trainModelChunks function is designed to handle training the GAN model in smaller, manageable chunks to avoid RAM overflow and ensure training can resume from checkpoints if interrupted.

1. Initialize Variables: Calculate the total number of chunks and initialize the overall loss history.
2. Chunk Loop: Loop through each chunk of the total training epochs.
3. RAM Monitoring: Check RAM usage to avoid memory overflow.
4. Chunk Training: For each chunk:
    - Train GAN: Call the trainGan function to train the model for the specified chunk size.
    - Save Checkpoints: Save the model weights after each chunk to ensure progress is not lost.
    - Generate and Save Images: Generate and save images to visualize training progress.
    - Calculate FID Score: Evaluate the model using FID score after each chunk.
5. Clear Session: Clear the TensorFlow session to free up memory.
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
```
### Build Model and Configuration
In this section, we build the generator and discriminator models, configure their training setup, and prepare the GAN model for training.

First, we define and initialize the generator and discriminator models. Summaries of both models are printed to provide an overview of their architectures. Next, we check for any available model checkpoints to potentially continue training from a saved state. If a checkpoint is found, the model weights are loaded accordingly; otherwise, the models are initialized from scratch.

To optimize the training process, we use the Adam optimizer with specific learning rates and beta values. The discriminator is compiled separately to allow for independent training before combining it with the generator to create the complete GAN model. The GAN model is then compiled using binary cross-entropy loss.
```python

# Build Model and Configuration
# Build model
inputDim = 100
pixel = 64
generator = buildGenerator(inputDim)
discriminator = buildDiscriminator(pixel)

# Generator Summary
generator.summary()

# Discriminator Summary
discriminator.summary()

# Load Latest Checkpoint
lastModelCheckPoint = latestCheckpoint(Path().cwd())
# lastModelCheckPoint = latestCheckpoint('/kaggle/input/dcgan/tensorflow1/default/1')

if lastModelCheckPoint is not None:
    print(f"Continue from latest checkpoint: {lastModelCheckPoint}")

    # Load only weights
    generator.load_weights(lastModelCheckPoint)
    discriminator.load_weights(lastModelCheckPoint.replace('generator', 'discriminator'))
else:
    print("No Checkpoint found, start model from beginning")

# Initialize the Adam optimizer
generatorOptimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminatorOptimizer = Adam(learning_rate=0.0001, beta_1=0.3)

# Compile discriminator independently
discriminator.compile(loss='binary_crossentropy', optimizer=discriminatorOptimizer, metrics=['accuracy'])

# Freeze discriminator when creating GAN
discriminator.trainable = False

# Create the GAN model
gan_input = Input(shape=(inputDim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Compile GAN model
gan.compile(loss='binary_crossentropy', optimizer=generatorOptimizer)

gan.summary()
```
### Model Training
Given the limitations of available RAM and the frequent crashes caused by the model's high RAM requirements, it is essential to use checkpoints and train in batches. This approach ensures that if a crash occurs, the training process can resume from the last saved checkpoint, preventing the loss of progress. We are using a total of 50 epochs, and training is conducted one epoch at a time.

To train the model, the following steps are implemented:
- Clear Session: Ensure a clean state by clearing any previous session.
- Set Seeds: Set random seeds for reproducibility.
- Train Data: Initialize training with the desired number of epochs and chunk size.
- Training Process: Use a device-specific context (/GPU:0) to leverage GPU acceleration for training the GAN model in chunks.
```python
# Clear Session
seed = 20
tf.keras.backend.clear_session()
np.random.seed(seed)
tf.random.set_seed(seed)

# Train Data
targetEpochs = 50
chunkSize = 1
currentEpochs = getNextFileNumber('generated_anime_faces', 'generator_epoch_', '.png')
epochs = targetEpochs - currentEpochs

with tf.device('/GPU:0'):
    history = trainModelChunks(gan, generator, discriminator, trainData,
                               totalEpochs=epochs, chunkSize=chunkSize, batchSize=128)
```
### Model Evaluation
In this section, we evaluate the performance of the trained GAN model by analyzing the loss trends and visually inspecting the generated images. Additionally, we calculate the Fréchet Inception Distance (FID) score to quantitatively assess the quality of the generated images.

1. Plot Model Loss: We plot the loss curves for both the discriminator and generator to understand their training progress over the epochs.
2. Generate and Display Images: We generate and display a grid of images to visually inspect the quality of the images produced by the GAN.
3. Calculate FID Score: The FID score is computed to evaluate the similarity between the real and generated images. A lower FID score indicates higher similarity.

Here is the code for evaluating the model:
```python
# Plot Model Loss
fig, ax = plt.subplots(ncols=2, figsize=(16, 5))

# Plot Discriminator Loss
ax[0].plot(range(1, len(history['dLoss']) + 1), history['dLoss'], label='Discriminator Loss', color='blue')
ax[0].set_title('Discriminator Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot Generator Loss
ax[1].plot(range(1, len(history['gLoss']) + 1), history['gLoss'], label='Generator Loss', color='orange')
ax[1].set_title('Generator Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

# Show plot
plt.tight_layout()
plt.show()

# Function to generate and display images
def generateAndDisplayImages(generator, noiseDim=100, numImages=16):
    # Generate noise
    noise = np.random.normal(0, 1, (numImages, noiseDim))

    # Generate images
    genImgs = generator(noise, training=False)

    # Convert Tensor to NumPy array
    genImgs = genImgs.numpy()

    # Rescale images to [0, 1]
    genImgs = 0.5 * genImgs + 0.5

    # Convert to uint8 for compatibility with matplotlib
    genImgs = (genImgs * 255).astype(np.uint8)

    # Plot the generated images
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    count = 0
    for i in range(4):
        for j in range(4):
            # Handle grayscale or RGB
            if genImgs.shape[-1] == 1:  # Grayscale
                axs[i, j].imshow(genImgs[count, :, :, 0], cmap='gray')
            else:  # RGB
                axs[i, j].imshow(genImgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.show()

# Display generated images
generateAndDisplayImages(generator, noiseDim=100, numImages=32)

# Calculate FID score
realImagesBatch = next(iter(trainData.take(1)))
realImagesBatch = 0.5 * (realImagesBatch + 1)
noiseForFid = np.random.normal(0, 1, (realImagesBatch.shape[0], 100))
generatedImagesForFid = generator.predict(noiseForFid)
fidScore = calculateFid(model, realImagesBatch, generatedImagesForFid)
print(fidScore)
```
Note:
From the evaluation results, the GAN-generated images closely resemble the images in the dataset used. However, the colors in the images appear slightly faded. This fading could be due to several reasons, such as the model's tendency to average pixel values, which may affect the model's ability to capture vibrant colors accurately. The FID (Fréchet Inception Distance) score remained stable at 5000 from the beginning to the end of the training process, indicating consistent performance throughout the training period.
