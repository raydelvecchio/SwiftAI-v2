from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from data.corpus import get_json_filenames, clean_song
import json
import torch


class LyricLines(Dataset):
    """
    Pytorch dataset class for our lines of lyrics. Contains a tokenizer which tokenizes in the style of GPT2 to train.
    Default max length of 20 words per line.
    Saves masks so we can ignore our padding during training!
    Docs for tokenizer found here:
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
    """

    def __init__(self, lyrics_lines: list, max_len=20, eos_token='<|endoftext|>'):
        self.lines = lyrics_lines
        self.num_lines = len(self.lines)
        self.tokenizer = get_special_tokenizer()

        print("Tokenizing lyric lines data...")
        self.input_ids = []
        self.attention_masks = []
        for line in self.lines:
            # tokenizes line between beginning EOS tokens, since original GPT2 model did this with text
            line_bos_eos = f'{eos_token} {line} {eos_token}'
            tokenizer_out = self.tokenizer(line_bos_eos, max_length=max_len, truncation=True, padding='max_length')
            line_tokens = tokenizer_out['input_ids']
            mask = tokenizer_out['attention_mask']
            self.input_ids.append(torch.Tensor(line_tokens))
            self.attention_masks.append(torch.Tensor(mask))
        print("Tokenized!\n")

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_masks[index]


class LyricSongs(Dataset):
    """
    Pytorch dataset class for training on songs rather than lines.
    Contains a tokenizer which tokenizes in the style of GPT2 to train.
    Default max length per line of 500 words per song! This is subject to change however.
    Saves masks so we can ignore our padding during training!
    Docs for tokenizer found here:
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
    """

    def __init__(self, songs: list, max_len=500, eos_token='<|endoftext|>'):
        self.songs = songs
        self.num_songs = len(self.songs)
        self.tokenizer = get_special_tokenizer()

        print("Tokenizing songs data...")
        self.input_ids = []
        self.attention_masks = []
        for song in self.songs:
            # tokenizes line between beginning EOS tokens, since original GPT2 model did this with text
            song_bos_eos = f'{eos_token} {song} {eos_token}'
            tokenizer_out = self.tokenizer(song_bos_eos, max_length=max_len, truncation=True, padding='max_length')
            song_tokens = tokenizer_out['input_ids']
            mask = tokenizer_out['attention_mask']
            self.input_ids.append(torch.Tensor(song_tokens))
            self.attention_masks.append(torch.Tensor(mask))
        print("Tokenized!\n")

    def __len__(self):
        return self.num_songs

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_masks[index]


def get_special_tokenizer(pad_token='<|pad|>'):
    """
    Creates tokenizer with special tokens if we need. Currently deprecated and not used in favor of the direct GPT2
    Tokenizer without special tokens.
    """
    return GPT2Tokenizer.from_pretrained('gpt2', pad_token=pad_token)


def clean_line(line: str) -> str:
    """
    Cleans a line of a song per our preprocessing spec. May have to change depending on the spec of our pretrained
    model we want to fine tune.
    """
    if "LiveGet" in line:
        return ""
    for c in ['\"', '(', ')']:
        line = line.replace(c, "")
    line = line.replace('\n\n\n', '\n\n')
    line = line.replace('\n\n', '\n')
    line = line.replace('\n', '')
    return line


def preprocess_get_dataset_and_tokenizer(songs_over_lines=True, eos_token='<|endoftext|>') -> (Dataset, GPT2Tokenizer):
    """
    Preprocesses our generated .txt file to remove punctuation, parenthesis, unwanted lines (like LiveGet ticket ads),
    and more. Returns Dataset object containing our lyric lines, and tokenizer object to access tokenizations. To
    create a dataset from lines of Taylor Swift, we'd need to have our corpus created in corpus.py first.

    Two options for preprocess: we can 1.) feed our model a series of song lines or 2.) feed our model a series of
    songs themselves. We can adjust this with the songs_over_lines switch.
    """
    if songs_over_lines:
        songs = []
        album_files = get_json_filenames('data')
        for album in album_files:
            album = f'data/{album}'
            with open(album) as lyr:
                j = json.load(lyr)
            tracks = j['tracks']
            for track in tracks:
                lyrics = track['song']['lyrics']
                lyrics = clean_song(lyrics)
                songs.append(lyrics)
        dataset = LyricSongs(songs)
        return dataset, dataset.tokenizer
    else:
        with open('data/lyrics.txt', 'r', encoding="utf-8") as f:
            lines = f.readlines()
        lines = [clean_line(line) for line in lines]
        dataset = LyricLines(lines)
        return dataset, dataset.tokenizer


if __name__ == "__main__":
    preprocess_get_dataset_and_tokenizer()
