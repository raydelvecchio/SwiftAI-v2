import torch


class SwiftAI:
    def __init__(self):
        self.model = torch.load('saved_vars/swiftai_model.pth')


if __name__ == "__main__":
    swift = SwiftAI()
