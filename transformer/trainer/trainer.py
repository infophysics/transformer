import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import os
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from transformer.losses.loss import Loss


class Trainer:
    """
    """
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        self.parse_config()
    
    def parse_config(self):
        self.experiment_name = self.config['experiment_name']
        self.train_val_split = self.config["train_val_split"]
        self.seq_len = self.config["seq_len"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.epochs = self.config["epochs"]
        self.lr = self.config['lr']
        self.model_folder = self.config["model_folder"]
        self.datasource = self.config["datasource"]
        self.model_basename = self.config["model_basename"]
        self.preload = self.config["preload"]

    def construct_dataloaders(
        self,
        dataset
    ):
        train_ds_size = int(0.9 * len(dataset))
        val_ds_size = len(dataset) - train_ds_size
        train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])
        self.train_dataloader = DataLoader(
            train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_dataloader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers
        )

    def train(
        self,
        dataset,
        model
    ):
        self.construct_dataloaders(dataset)
        # Define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        if (device == 'cuda'):
            print(f"Device name: {torch.cuda.get_device_name(device.index)}")
            print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        device = torch.device(device)
        # Make sure the weights folder exists
        Path(f"{self.datasource}_{self.model_folder}").mkdir(parents=True, exist_ok=True)
        # Tensorboard
        writer = SummaryWriter(self.experiment_name)

        loss_fn = Loss(
            self.val_dataloader.dataset.source_tokenizer.token_to_id('[PAD]'),
            self.val_dataloader.target_tokenizer.get_vocab_size(),
            device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, eps=1e-9)

        # If the user specified a model to preload before training, load it
        initial_epoch = 0
        global_step = 0
        total_iterations = self.epochs * len(self.train_dataloader)
        if self.preload == 'latest':
            model_filename = model.latest_weights_file_path(
                f"{self.datasource}_{self.model_folder}",
                f"{self.model_basename}*"
            )
        elif self.preload is not None:
            model_filename = model.get_weights_file_path(
                f"{self.datasource}_{self.model_folder}",
                f"{self.model_basename}{self.preload}"
            )
        else:
            model_filename = None
        if model_filename:
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print('No model to preload, starting from scratch')

        for epoch in range(initial_epoch, self.epochs):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(
                self.train_dataloader, 
                desc=f"Processing Iter {global_step}/{total_iterations} - Epoch {epoch:02d}"
            )
            for batch in batch_iterator:

                batch = model(batch)
                # Compute the loss using a simple cross entropy
                loss = loss_fn(batch)
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Run validation at the end of every epoch
            self.run_validation(
                model,
                device, 
                lambda msg: batch_iterator.write(msg),
                global_step,
                writer
            )

            # Save the model at the end of every epoch
            model_filename = model.get_weights_file_path(
                f"{self.datasource}_{self.model_folder}",
                f"{self.model_basename}{epoch:02d}"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
    
    def greedy_decode(
        self,
        data,
        model,
        device
    ):
        sos_idx = self.val_dataloader.dataset.target_tokenizer.token_to_id('[SOS]')
        eos_idx = self.val_dataloader.dataset.target_tokenizer.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        data = model.embed(data)
        data = model.pos_encode(data)
        data = model.encode(data)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(data['token_embedding']).to(device)
        while True:
            if decoder_input.size(1) == self.seq_len:
                break

            # build mask for target
            decoder_mask = self.val_dataloader.dataset.causal_mask(decoder_input.size(1)).type_as(data['source_mask']).to(device)

            # calculate output
            out = model.decode(data)

            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(data['token_embedding']).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
    
    def run_validation(
        self,
        model,
        device, 
        print_msg, 
        global_step, 
        writer, 
        num_examples=2
    ):
        model.eval()
        count = 0
        source_texts = []
        expected = []
        predicted = []
        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in self.val_dataloader:
                count += 1
                model_out = model.greedy_decode(batch, model)

                source_text = batch["source_text"][0]
                target_text = batch["target_text"][0]
                model_out_text = self.val_dataloader.dataset.target_tokenizer.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                
                # Print the source, target and model output
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg('-'*console_width)
                    break
        
        if writer:
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()