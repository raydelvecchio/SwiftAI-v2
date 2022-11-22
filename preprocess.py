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

    def __init__(self, lyrics_lines: list, max_len=20, unk_token='<|unk|>', bos_token='<|start|>',
                 eos_token='<|newline|>', pad_token='<|pad|>'):
        self.lines = lyrics_lines
        self.num_lines = len(self.lines)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', unk_token=unk_token, bos_token=bos_token,
                                                       eos_token=eos_token, pad_token=pad_token, add_prefix_space=True)

        print("Tokenizing lyric lines data...")
        self.input_ids = []
        for line in self.lines:
            # tokenizes line between beginning of sentence token and end of sentence token
            line_bos_eos = f'{bos_token} {line} {eos_token}'
            # pads all sentences to same length (max length of line)
            line_tokens = self.tokenizer(line_bos_eos, max_length=max_len, padding='max_length',
                                         return_token_type_ids=False)['input_ids']
            # converts tokens to a tensor and pads to list of line ids
            self.input_ids.append(torch.Tensor(line_tokens))
        print("Tokenized!")

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        return self.input_ids[index]


def clean_line(line: str) -> str:
    """
    Cleans a line of a song per our preprocessing spec. May have to change depending on the spec of our pretrained
    model we want to fine tune.
    """
    if "LiveGet" in line:
        return ""
    for c in ['\"', '\'', ':', ',', '.', '?', '!', ';', '(', ')', '']:
        line = line.replace(c, "")
    line = line.replace('-', " ")
    line = line.replace('\n\n\n', '\n\n')
    line = line.replace('\n\n', '\n')
    line = line.replace('\n', '')
    line = line.lower()
    return line


def preprocess_and_get_dataset() -> Dataset:
    """
    Preprocesses our generated .txt file to remove punctuation, parenthesis, unwanted lines (like LiveGet ticket ads),
    and more. Returns Dataset object containing our lyric lines!

    Two options for preprocess: we can 1.) feed our model a series of song lines or 2.) feed our model a series of
    songs themselves. We'll first try feeding it a bunch of song lines.
    """
    with open('data/lyrics.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [clean_line(line) for line in lines]
    dataset = LyricLines(lines)
    return dataset
