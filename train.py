from transformers import GPT2Model, GPT2Config
from preprocess import preprocess


class SwiftAITrainer:
    def __init__(self):
        """
        Initializes the pre-trained GPT2 model from configuration.
        Config docs: https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.
        """
        config = GPT2Config.from_pretrained('gpt2')
        self.model = GPT2Model(config)
        self.data = preprocess()


if __name__ == "__main__":
    trainer = SwiftAITrainer()
