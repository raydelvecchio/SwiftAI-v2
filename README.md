# SwiftAI V2!
This is the second iteration of SwiftAI! The first time, we used a 7-gram model with an RNN to make predictions and
generate songs. Now, with more NLP knowledge, we can up the ante and make the best model possible.

# Improvements on V1
As you know this project is the second version of my ill-fated SwiftAI V1. I've made the following enhancements on the
first version, listed here:
* Larger train dataset, including all "bonus" versions of all albums (deluxe, platinum, etc) and her new album, Midnights
* Fine-tuning a pre-trained model rather than building my own model
* Code improvements to make it cleaner, easier to read, and more efficient
* Huggingface/Pytorch instead of Tensorflow

# constants.py
File containing all constants necessary to train the model. Genius secrets/tokens stored as environment variables.

# download_data.py
Methods used to download lyric data from Genius. Should all be run locally from this file; methods never called
externally.

# preprocess.py
Methods used to generate corpus from .json lyrics files. `generate_corpus` should only be run locally for the purpose
of synthesizing a corpus from each album's lyrics. `preprocess` method is called externally to get data to train/test
our model, and all other methods support it.

# TODOs:
* get train/test labels/dictionary according to GPT2 tokenizer?
  * This also involves renaming/overriding the `preprocess` method to be `preformat`, since preprocess should
  return our training data/labels according to GPT2's tokenizer
* tokenize according to GPT2 tokenizer from huggingface
* fine tune GPT2 and save the model to make predictions
* upload to server to host a website for making such predictions
