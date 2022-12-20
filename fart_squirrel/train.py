"""
    Useful for training shadow models
"""
import torch as ch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager


from mico_competition import MLP


def accuracy(preds: ch.Tensor, labels: ch.Tensor) -> float:
    return (preds == labels).mean()


def train(model: nn.Module,
          train_loader: DataLoader,
          criterion,
          optimizer: optim.Optimizer,
          batch_size: int):
    model.train()

    losses = []
    top1_acc = []

    disable_dp = True
    max_physical_batch_size = 128
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        if disable_dp:
            data_loader = train_loader
        else:
            data_loader = memory_safe_data_loader

        # BatchSplittingSampler.__len__() approximates (badly) the length in physical batches
        # See https://github.com/pytorch/opacus/issues/516
        # We instead heuristically keep track of logical batches processed
        logical_batch_len = 0
        for i, (inputs, target) in enumerate(data_loader):
            inputs, target = inputs.cuda(), target.cuda()

            logical_batch_len += len(target)
            if logical_batch_len >= batch_size:
                logical_batch_len = logical_batch_len % max_physical_batch_size

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()
    return model


def n_datasets_generator(data, member, n_times, n_sample):
    for _ in range(n_times):
        indices = np.random.choice(len(data[0]), n_sample, replace=False)
        dataset_without = TensorDataset(data[0][indices], data[1][indices])
        X_wanted = ch.cat((data[0][indices], member[0].view(1, -1)))
        Y_wanted = ch.cat((data[1][indices], member[1].view(1)))
        dataset = TensorDataset(X_wanted, Y_wanted)
        yield dataset, dataset_without


def train_model(loader):
    batch_size = 512
    learning_rate = 0.001
    lr_scheduler_step = 5
    lr_scheduler_gamma = 0.9
    num_epochs = 30

    model = MLP()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_rate = 1 / len(loader)
    num_steps = int(len(loader) * num_epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

    for _ in range(num_epochs):
        train(model, loader, criterion, optimizer, batch_size)
        scheduler.step()
    
    return model
