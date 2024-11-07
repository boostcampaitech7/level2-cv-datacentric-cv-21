from torch.optim import lr_scheduler

def sched(args, optimizer):
    if args.scheduler == "multistep":
        return lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch // 2, args.max_epoch // 2 * 2], gamma=0.1)
    elif args.scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0)