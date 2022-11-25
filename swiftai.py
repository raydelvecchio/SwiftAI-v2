import torch
from train import SwiftAITrainer
from preprocess import get_special_tokenizer


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

    def write_song(self, text_prompt: str, length=300, k=50, p=0.9, max_temp=1.5, num_ret=3, ngram_block=12) -> list:
        """
        Given a text prompt, generate text with our model! Hyperparameters like max length, k, p, and temp can be
        adjusted to vary the generation of text that we produce. In the future, we could do many samples with a lot of
        k, p, and temp and compare it to a metric to find the "best" generation. Generates num_ret outputs all with
        decreasing temperature values for the highest variance of answers! Can block out repeating grams with the
        repeat_block variable.
        Example text generation: https://huggingface.co/blog/how-to-generate
        Generate function docs: https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
        """
        self.model.eval()

        starting_tokens = self.tokenizer.encode(text_prompt, return_tensors='pt').to(self.device)

        # force_tokens = starting_tokens.tolist()  # we want to retain context here, so force model to generate them

        outputs = []
        max, end, inc = int(max_temp * 10), int((max_temp - (0.1 * (num_ret + 1))) * 10), int(-0.1 * 10)
        for i in range(max, end, inc):
            outputs += self.model.generate(starting_tokens, do_sample=True, top_k=k, top_p=p, max_length=length,
                                           temperature=i / 10, no_repeat_ngram_size=ngram_block)
        return [self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs]
