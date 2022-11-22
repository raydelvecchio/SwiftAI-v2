from transformers import GPT2LMHeadModel, GPT2Config, get_cosine_with_hard_restarts_schedule_with_warmup
from preprocess import preprocess_get_dataset_and_tokenizer
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
import torch


class SwiftAITrainer:
    def __init__(self, use_gpu=True, train_size=0.8, batch_size=1, learning_rate=1e-3, epochs=3, warmup_steps=1e3,
                 lr_cycles=4):
        """
        Initializes the pre-trained GPT2 model from configuration, resizes embedding to include new vocabulary created
        (such as start, pad, newline, UNK, etc), creates training and validation dataloaders for training, defines
        optimizer, and defines scheduler to use to adjust learning rate over time. I use a cosine restarting warmup
        schedule to define how my learning rate changes as we train. To achieve a higher accuracy, we can adjust the
        input hyperparameters until we converge to a better model. If this doesn't work, we can examine docs for
        our optimizer and scheduler to change default values if need be. Can opt to use GPU for training or stick
        with CPU usage only with use_gpu parameter.
        Config docs: https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.
        Pre-Model docs: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.
        Pytorch Module docs: https://pytorch.org/docs/stable/generated/torch.nn.Module.html.
        Dataloader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
        AdamW docs: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW.
        Scheduler docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType.
        GPT2LMHead docs: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.
        """
        print("Creating all training objects and variables...")
        self.epochs = epochs
        self.use_gpu = use_gpu

        if use_gpu:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            generator = torch.Generator(device="cuda")
        else:
            generator = torch.Generator(device="cpu")
        lines_dataset, tokenizer = preprocess_get_dataset_and_tokenizer()
        train_data, validation_data = random_split(lines_dataset,
                                                   [math.floor(train_size * len(lines_dataset)),
                                                    len(lines_dataset) - math.floor(train_size * len(lines_dataset))],
                                                   generator=generator)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                       generator=generator)
        self.validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                            generator=generator)

        config = GPT2Config.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel(config)
        self.model.resize_token_embeddings(len(tokenizer))
        if use_gpu:
            self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                                            num_warmup_steps=warmup_steps,
                                                                            num_training_steps=epochs * len(
                                                                                self.train_loader),
                                                                            num_cycles=lr_cycles)
        print("SwiftAITrainer ready to train!\n")
        self.save_model("untrained_swiftai")

    def save_model(self, name: str):
        """
        Saves both model weights and model object to reload later at any time we want.
        Docs: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model.
        The folder "saved_vars" is in the gitignore because our model/weights are massive (370 MB of numbers)!
        """
        folder = "saved_vars/"
        torch.save(self.model.state_dict(), f'{folder}{name}_weights.pth')
        torch.save(self.model, f'{folder}{name}_model.pth')

    def train(self, save_model_end=True, save_model_epoch=False, plot_loss=False):
        """
        Trains model on our initialized data loaders! After training is complete, we save the model and its weights to
        be loaded by our predictor in swiftai.py. Also returns the model! Can set parameters to determine when we
        save the model during training, if at all.
        """
        self.model.train()

        loss_list = []

        for epoch in range(self.epochs):
            print(f'Training Epoch {epoch + 1}...')
            for batch in self.train_loader:
                if self.use_gpu:
                    inputs = batch.cuda().long()
                else:
                    inputs = batch.long()

                # grads zeroed; Pytorch accumulates them, so we must reset lest gradients be influenced by prev grads
                self.model.zero_grad()
                self.optimizer.zero_grad()

                # forward[0] is loss Tensor of next token prediction, forward[1] is logits Tensor
                # inputs and labels are the same since we're using GPT2LMHead Model, which creates labels from inputs
                outputs = self.model.forward(inputs, labels=inputs)
                loss = outputs[0]
                loss_val = float(loss.item())
                loss_list.append(loss_val)
                print(f'Loss at batch {len(loss_list)}: {loss_val}')

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if save_model_epoch:
                self.save_model(f'epoch_{epoch + 1}_swiftai')

        if save_model_end:
            self.save_model("trained_swiftai")

        if plot_loss:
            plt.plot(loss_list)
            plt.ylabel("Loss Value")
            plt.xlabel("Batch Number")
            plt.show()

        return self.model


if __name__ == "__main__":
    trainer = SwiftAITrainer()
    trainer.train(plot_loss=True)
