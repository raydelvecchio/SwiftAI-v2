# SwiftAI V2!
This is the second iteration of SwiftAI! The first time, we used a 7-gram model with an RNN to make predictions and
generate songs. Now, with more NLP knowledge, we can up the ante and make the best model possible. Uses [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#openai-gpt2).

# Improvements on V1
As you know this project is the second version of my ill-fated SwiftAI V1. I've made the following enhancements on the
first version, listed here:
* Larger train dataset, including all "bonus" versions of all albums (deluxe, platinum, etc) and her new album, Midnights
* Fine-tuning a pre-trained model rather than building my own model
* Code improvements to make it cleaner, easier to read, and more efficient
* Huggingface/Pytorch instead of Tensorflow

# Notes on Running and Installation
This codebase, specifically when training the model, requires CUDA (and Pytorch CUDA). You can check your CUDA version
with `nvcc --version`. You can then check how to install Pytorch CUDA [here](https://pytorch.org/get-started/locally/). I 
used the following command: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117`.

# constants.py
File containing all constants necessary to train the model. Genius secrets/tokens stored as environment variables.

# data/download_data.py
Methods used to download lyric data from Genius. Should all be run locally from this file; methods never called
externally.

# data/corpus.py
Creates our text corpus of Taylor Swift songs. Should only be run once locally after downloading data. May be deprecated
in the future if we choose to train on songs rather than lines.

# preprocess.py
Methods and classes used to generate PyTorch dataset from our generated corpus.

# train.py
Contains the SwiftAITrainer class, which is used to train and save the model. This class creates all necessary variables
for Pytorch training, then uses them to fine tune the GPT2 model. 

# swiftai.py
Contains the SwiftAI class! This class imports the model and then uses it to make predictions on new song lyrics.

# ERRORS and FIXES
* model.forward() input cannot be a Tensor
  * Had to convert input and labels to Long datatype before passing into model.forward()
* CUDA Not Working, receiving `philox_cuda_state for an unexpected CUDA generator used during capture. In regions captured by CUDA graphs, you may only use the default CUDA RNG generator on the device that's current when capture begins. If you need a non-default (user-supplied) generator, or a generator on another device, please file an issue` error when calling forward pass in train loop
  * Could try running on CPU instead? Or look up how to properly set up CUDA?
  * Could it be the tokens we added that's throwing this off?

# TODOs:
* in our dataset, we could get mask for what we padded to our sentence?
* Train model with metrics to track progress (loss, accuracy, BLEU)
  * As part of this, we should save model to a file to load up at any other time
* Load saved model and make predictions based on it (new file separate from train.py)
* upload to server to host a website for making such predictions
