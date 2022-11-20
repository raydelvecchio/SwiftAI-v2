from transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class LyricLines(Dataset):
    """
    Pytorch dataset class for our lines of lyrics. Contains a tokenizer which tokenizes in the style of GPT2 to train.
    Docs for tokenizer found here:
    https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
    """
    def __init__(self, lyrics_lines: list, max_len=1024, unk_token='<|unk|>', bos_token='<|startoftext|>',
                 eos_token='<|newline|>'):
        self.lines = lyrics_lines
        self.num_lines = len(self.lines)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', unk_token=unk_token, bos_token=bos_token,
                                                       eos_token=eos_token, add_prefix_space=True)

        print("Beginning to tokenize lyric lines data...")
        self.input_ids = []
        for line in self.lines:
            # tokenizes line between beginning of sentence token and end of sentence token
            self.input_ids.append(self.tokenizer(f'{bos_token} {line} {eos_token}', max_length=max_len)['input_ids'])

    def __len__(self):
        return self.num_lines

    def __getitem__(self, item):
        return self.input_ids[item]

    def get_tokenizer(self):
        return self.tokenizer


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
    return line


def preprocess():
    """
    Preprocesses our generated .txt file to remove punctuation, parenthesis, unwanted lines (like LiveGet ticket ads),
    and more. Should return our preprocessed data in string format.

    Two options for preprocess: we can 1.) feed our model a series of song lines or 2.) feed our model a series of
    songs themselves. We'll first try feeding it a bunch of song lines.
    """
    with open('lyrics.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [clean_line(line) for line in lines]
    dataset = LyricLines(lines)
    return dataset


if __name__ == "__main__":
    preprocess()
