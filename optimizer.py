import torch

def optim(optimizer_name, learning_rate, trainable_params):
    return getattr(torch.optim, optimizer_name)(trainable_params, lr=learning_rate)
