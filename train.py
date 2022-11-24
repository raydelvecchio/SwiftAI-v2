from transformers import GPT2LMHeadModel, get_cosine_with_hard_restarts_schedule_with_warmup
from preprocess import preprocess_get_dataset_and_tokenizer
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
import torch


def print_cuda_memory_info():
    """
    Prints memory information for GPU; used for debugging out of memory errors.
    """
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


class SwiftAITrainer:
    def __init__(self, use_gpu=True, load_untrained=False, untrained_path=None, train_size=0.99, batch_size=16,
                 learning_rate=1e-3, epochs=3, warmup_steps=4*1e2, lr_cycles=3, songs_over_lines=True):
        """
        Initializes the pre-trained GPT2 model from configuration, resizes embedding to include new vocabulary created
        (such as start, pad, newline, UNK, etc), creates training and validation dataloaders for training, defines
        optimizer, and defines scheduler to use to adjust learning rate over time. I use a cosine restarting warmup
        schedule to define how my learning rate changes as we train. To achieve a higher accuracy, we can adjust the
        input hyperparameters until we converge to a better model. If this doesn't work, we can examine docs for
        our optimizer and scheduler to change default values if need be. Can opt to use GPU for training or stick
        with CPU usage only with use_gpu parameter. Can also load untrained model instead of downloading it every
        time by setting load_untrained=True and specifiying an untrained_path.

        Config docs: https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.
        Pre-Model docs: https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.
        Pytorch Module docs: https://pytorch.org/docs/stable/generated/torch.nn.Module.html.
        Dataloader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
        AdamW docs: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW.
        Scheduler docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType.
        GPT2LMHead docs: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.

        For songs, a batch size of 1 works the best locally. For lines, a batch size of 16 works best.
        """
        self.songs_over_lines = songs_over_lines

        print("Creating all training objects and variables...")
        self.epochs = epochs
        self.use_gpu = use_gpu

        if use_gpu:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            generator = torch.Generator(device="cuda")
        else:
            generator = torch.Generator(device="cpu")

        dataset, tokenizer = preprocess_get_dataset_and_tokenizer(songs_over_lines=songs_over_lines)

        train_data, validation_data = random_split(dataset,
                                                   [math.floor(train_size * len(dataset)),
                                                    len(dataset) - math.floor(train_size * len(dataset))],
                                                   generator=generator)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                       generator=generator)
        self.validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                            generator=generator)

        if load_untrained and untrained_path is not None:
            print("Loading saved untrained model...")
            self.model = torch.load(untrained_path)
            print("Model loaded!\n")
        else:
            print("Downloading GPT2 weights...")
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            print("Loaded!\n")

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

        if not load_untrained:
            print("Saving untrained model...")
            self.save_model("untrained_swiftai")
            print("Model saved!\n")

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
        if self.use_gpu:
            torch.cuda.empty_cache()

        self.model.train()

        loss_list = []

        for epoch in range(self.epochs):
            print(f'\nTraining Epoch {epoch + 1}...\n')
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
                if self.songs_over_lines:
                    append = "songs"
                else:
                    append = "lines"
                print("Saving model after epoch...")
                self.save_model(f'epoch_{epoch + 1}_swiftai_{append}')
                print("Model saved!\n")

        if save_model_end:
            if self.songs_over_lines:
                append = "songs"
            else:
                append = "lines"
            print("Saving model after training...")
            self.save_model(f'trained_swiftai_{append}')
            print("Model saved!\n")

        if plot_loss:
            plt.plot(loss_list)
            plt.ylabel("Loss Value")
            plt.xlabel("Batch Number")
            plt.show()

        return self.model


if __name__ == "__main__":
    trainer = SwiftAITrainer(load_untrained=True, untrained_path='saved_vars/untrained_swiftai_model.pth',
                             songs_over_lines=True, batch_size=1)
    trainer.train(plot_loss=True)
