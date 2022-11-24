from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch


class LyricLines(Dataset):
    """
    Pytorch dataset class for our lines of lyrics. Contains a tokenizer which tokenizes in the style of GPT2 to train.
    Docs for tokenizer found here:
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
    Default max length per line of 20 words.
    """

    def __init__(self, lyrics_lines: list, max_len=20, eos_token='<|endoftext|>'):
        self.lines = lyrics_lines
        self.num_lines = len(self.lines)
        self.tokenizer = get_special_tokenizer()

        print("Tokenizing lyric lines data...")
        self.input_ids = []
        for line in self.lines:
            # tokenizes line between beginning EOS tokens, since original GPT2 model did this with text
            line_bos_eos = f'{eos_token} {line} {eos_token}'
            line_tokens = self.tokenizer(line_bos_eos, max_length=max_len, truncation=True,
                                         padding='max_length')['input_ids']
            self.input_ids.append(torch.Tensor(line_tokens))
        print("Tokenized!\n")

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        return self.input_ids[index]


def get_special_tokenizer(pad_token='<|pad|>'):
    """
    Creates tokenizer with special tokens if we need. Currently deprecated and not used in favor of the direct GPT2
    Tokenizer without special tokens.
    """
    return GPT2Tokenizer.from_pretrained('gpt2', pad_token=pad_token)


def clean_line(line: str, max_len=20) -> str:
    """
    Cleans a line of a song per our preprocessing spec. May have to change depending on the spec of our pretrained
    model we want to fine tune.
    """
    if "LiveGet" in line or len(line.split()) >= max_len:
        return ""
    for c in ['\"', '(', ')']:
        line = line.replace(c, "")
    line = line.replace('\n\n\n', '\n\n')
    line = line.replace('\n\n', '\n')
    line = line.replace('\n', '')
    return line


def preprocess_get_dataset_and_tokenizer() -> (Dataset, GPT2Tokenizer):
    """
    Preprocesses our generated .txt file to remove punctuation, parenthesis, unwanted lines (like LiveGet ticket ads),
    and more. Returns Dataset object containing our lyric lines, and tokenizer object to access tokenizations.

    Two options for preprocess: we can 1.) feed our model a series of song lines or 2.) feed our model a series of
    songs themselves. We'll first try feeding it a bunch of song lines.
    """
    with open('data/lyrics.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [clean_line(line) for line in lines]
    print(len(lines))
    dataset = LyricLines(lines)
    return dataset, dataset.tokenizer
