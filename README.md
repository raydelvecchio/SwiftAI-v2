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
Methods and classes used to generate PyTorch dataset from our generated corpus. Can generate a dataset for training
either line-by-line or song-by-song!

# train.py
Contains the SwiftAITrainer class, which is used to train and save the model. This class creates all necessary variables
for Pytorch training, then uses them to fine tune the GPT2 model. Can train line by line or song by song. Tons of 
customizability for your model's training!

# swiftai.py
Contains the SwiftAI class! This class imports the model and then uses it to make predictions on new song lyrics.

# ERRORS and FIXES
* Train loop: model.forward() input cannot be a Tensor
  * Had to convert input and labels to Long datatype before passing into model.forward()
* Train loop: CUDA Not Working, receiving `philox_cuda_state for an unexpected CUDA generator used during capture. In regions captured by CUDA graphs, you may only use the default CUDA RNG generator on the device that's current when capture begins. If you need a non-default (user-supplied) generator, or a generator on another device, please file an issue` error when calling forward pass in train loop
  * Solved this by adding `generator=torch.Generator(device="cuda")` to DataLoader instantiation, but now I get `RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'` error
  * Ended up solving this by adding `generator=torch.Generator(device="cuda")` to `random_split()`, and adding `torch.set_default_tensor_type(torch.cuda.FloatTensor)` to the top of my `__init__()` method in the trainer class.
* Train loop: CUDA Cannot be initialized, receiving `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling 'cublasCreate(handle)'` error on train loop
  * This is probably because of my tokenizer being expanded to support special characters; I should do away with this?
  * Try running this on CPU to figure out the error and then switch back to GPU?
  * Best lead: [https://github.com/huggingface/transformers/issues/6263](https://github.com/huggingface/transformers/issues/6263)
* Train loop: `IndexError: index out of range in self`
  * Only happened when running on CPU, not GPU
  * Fixed by adding `self.tokenizer.add_special_tokens(DICT)` line in preprocess.py Dataset class
  * Also added `self.model.resize_token_embeddings(len(tokenizer))` in SwiftAITrainer class to account for new tokens
* Train loop with batch size >1: `RuntimeError: stack expects each tensor to be equal size, but got [20] at entry 0 and [22] at entry 7`
  * Does not occur when batch size is = 1; model trains in this case
  * Fixed by defining a pad token in our tokenizer and setting `max_length=max_len, truncation=True, padding='max_length'` when creating our initial tokenizations!
* Training on songs dataset, not lyrics: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 148.00 MiB (GPU 0; 4.00 GiB total capacity; 2.32 GiB already allocated; 0 bytes free; 3.35 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF`
  * Reducing to a batch size of 1 fixes this
  * Could also accumulate gradients or something to fix this in the future?
* Bad text generation produces a ton of <|EOS|> tokens and short songs (when trained on songs)
  * Fixed by only appending <|EOS|> before and after an entire SONG, rather than before and after each LINE
  * Now it generates entire songs!

# TODOs:
* Add or remove model validation during training
* Improve song generation:
  * Accumulate gradients to avoid out of memory issue?
  * Use masks during training to block out padded text
  * Look at other implementations of text generation?
  * Try to daisy chain generation together with each other?
  * Filter out `See Taylor Swift LiveGet tickets as low as $307You might also like` metadata line
* upload to server to host a website for making such predictions
