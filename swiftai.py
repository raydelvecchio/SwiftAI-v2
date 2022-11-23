import torch
from train import SwiftAITrainer
from preprocess import clean_line
from transformers import GPT2Tokenizer


class SwiftAI:
    def __init__(self, load_path: str, use_gpu=True, load_not_train=True):
        """
        Initializes variables needed to make predictions for new songs!
        :param load_not_train: If this is true, we load the model from the saved file. If false, we train it from
        scratch with the SwiftAITrainer class.
        :param use_gpu: makes predictions using GPU if true, makes predictions using CPU if false
        """
        if load_not_train:
            print("Loading model from path...")
            self.model = torch.load(load_path)
            print("Model loaded!")
        else:
            trainer = SwiftAITrainer()
            self.model = trainer.train(save_model_end=False)

        if use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def make_predictions(self, text_prompt: str, max_length=20) -> str:
        """
        Given a text prompt, generate text with our model!
        Example text generation: https://huggingface.co/blog/how-to-generate
        """
        text_prompt = clean_line(text_prompt)
        self.model.eval()
        starting_tokens = self.tokenizer.encode(text_prompt, return_tensors='pt').to(self.device)
        output = self.model.generate(starting_tokens, max_length=max_length)[0]  # best output is idx 1
        return self.tokenizer.decode(output)


if __name__ == "__main__":
    swift = SwiftAI('saved_vars/untrained_swiftai_model.pth')
    lines = swift.make_predictions("I love Antonio because he is")
    print(lines)
