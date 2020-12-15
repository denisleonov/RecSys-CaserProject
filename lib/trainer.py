import os
import lib
import time
import torch
import numpy as np

from tqdm import tqdm

from utils import *


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args):
        self.model = model
        self.train = train_data
        self.eval = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, use_cuda, k = args.k_eval)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        sequences_np = self.train.sequences.sequences
        targets_np = self.train.sequences.targets

        n_train = sequences_np.shape[0]

        print('total training instances: %d' % n_train)

        epoch_pbar = tqdm(range(0, self._n_iter), total=self._n_iter, desc='Train')
        for epoch_num in epoch_pbar:

            # set model to training mode
            self._net.train()

            if epoch_num % 5 == 0 or epoch_num == 0:
                sequences_np, targets_np = shuffle(sequences_np, targets_np)

                # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
                sequences, targets = (torch.from_numpy(sequences_np).long(),
                                      torch.from_numpy(targets_np).long())

                sequences, targets = (sequences.to(self._device),
                                      targets.to(self._device))
        
            mean_loss = self.train_epoch(sequences, targets, epoch_num)

            epoch_pbar.set_postfix_str('loss: ', mean_loss)


    def train_epoch(self, sequences, targets, epoch_num):
        self.model.train()
        #losses = []
        hidden = self.model.init_hidden()

        epoch_loss = 0.0
        batch_pbar = tqdm(enumerate(minibatch(sequences, targets, batch_size=self.batch_size)),
                          total=len(sequences) // self.batch_size, leave=True, desc=f'Epoch: {epoch_num + 1}')

        for (minibatch_num, (batch_sequences, batch_targets)) in batch_pbar:
            self.optim.zero_grad()

            hidden = hidden.detach()
            logit, hidden = self.model(batch_sequences, hidden)

            # output sampling
            logit_sampled = logit[:, batch_targets.view(-1)]
            loss = self.loss_func(logit_sampled)
            #losses.append(loss.item())
            epoch_loss += loss.item()

            batch_pbar.set_postfix_str('loss: ', loss.item())

            loss.backward()
            self.optim.step()

        
        return epoch_loss / (minibatch_num + 1)