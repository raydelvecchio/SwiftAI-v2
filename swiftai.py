import torch
from train import SwiftAITrainer


class SwiftAI:
    def __init__(self, load_not_train=True):
        """
        Initializes variables needed to make predictions for new songs!
        :param load_not_train: If this is true, we load the model from the saved file. If false, we train it from
        scratch with the SwiftAITrainer class.
        """
        if load_not_train:
            self.model = torch.load('saved_vars/swiftai_model.pth')
        else:
            trainer = SwiftAITrainer()
            self.model = trainer.train(save_model_end=False)


if __name__ == "__main__":
    swift = SwiftAI()
