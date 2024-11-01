import torch

def optim(args, trainable_params):
    return getattr(torch.optim, args.optimizer)(trainable_params, lr=args.learning_rate)