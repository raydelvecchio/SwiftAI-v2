import torch
from train import SwiftAITrainer
from preprocess import clean_line, get_special_tokenizer
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
            print("Model loaded!\n")
        else:
            trainer = SwiftAITrainer()
            self.model = trainer.train(save_model_end=False)

        if use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.tokenizer = get_special_tokenizer()

    def make_predictions(self, text_prompt: str, length=20, k=100, p=0.9, temp=1.2, num_ret=5) -> list:
        """
        Given a text prompt, generate text with our model! Hyperparameters like max length, k, p, and temp can be
        adjusted to vary the generation of text that we produce. In the future, we could do many samples with a lot of
        k, p, and temp and compare it to a metric to find the "best" generation.
        Example text generation: https://huggingface.co/blog/how-to-generate
        Generate function docs: https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        """
        self.model.eval()
        starting_tokens = self.tokenizer.encode(text_prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(starting_tokens, do_sample=True, top_k=k, top_p=p, max_length=length*10,
                                      temperature=temp, num_return_sequences=num_ret)
        return [self.tokenizer.decode(output) for output in outputs]


if __name__ == "__main__":
    swift = SwiftAI('saved_vars/trained_swiftai_model.pth')
    lines = swift.make_predictions("I've got a")
    print(lines)
