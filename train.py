from transformers import GPT2Model, GPT2Config, get_cosine_with_hard_restarts_schedule_with_warmup
from preprocess import preprocess
from torch.utils.data import DataLoader, random_split
import math
import torch


class SwiftAITrainer:
    def __init__(self, train_size=0.8, batch_size=8, learning_rate=1e-3, epochs=3, warmup_steps=1e3, lr_cycles=4):
        """
        Initializes the pre-trained GPT2 model from configuration, resizes embedding to include new vocabulary created
        (such as start, pad, newline, UNK, etc), creates training and validation dataloaders for training, defines
        optimizer, and defines scheduler to use to adjust learning rate over time. I use a cosine restarting warmup
        schedule to define how my learning rate changes as we train. To achieve a higher accuracy, we can adjust the
        input hyperparameters until we converge to a better model. If this doesn't work, we can examine docs for
        our optimizer and scheduler to change default values if need be.
        Config docs: https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.
        Model docs: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.
        Dataloader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
        AdamW docs: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW.
        Scheduler docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType.
        """
        print("Creating all training objects and variables...")
        self.epochs = epochs

        lines_dataset = preprocess()
        train_data, validation_data = random_split(lines_dataset,
                                                   [math.floor(train_size * len(lines_dataset)),
                                                    len(lines_dataset) - math.floor(train_size * len(lines_dataset))])
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True)

        self.device = torch.device('cuda')

        config = GPT2Config.from_pretrained('gpt2')
        self.model = GPT2Model(config)
        self.model.resize_token_embeddings(len(lines_dataset))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                                            num_warmup_steps=warmup_steps,
                                                                            num_training_steps=epochs * len(
                                                                                self.train_loader),
                                                                            num_cycles=lr_cycles)
        print("SwiftAITrainer ready to train!")


if __name__ == "__main__":
    trainer = SwiftAITrainer()
