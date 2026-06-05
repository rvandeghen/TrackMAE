import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset
from engine_for_pretraining_cotracker import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import wandb
from cuda_prefetcher import CUDAPrefetcher

# co-tracker
from cotrackerv3.models import CoTrackerFF

def get_args():
    parser = argparse.ArgumentParser('TrackMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--ratio_motion', default=0.15, type=float, nargs='+',)
    parser.add_argument('--mask_type_cotracker', default='normal',
                        type=str, nargs='+', help='masked strategy of video tokens/patches')
    parser.add_argument('--upsample', action='store_true',
                        help='use upsample in cotracker')
    
    parser.add_argument('--target_type', default='pixel', choices=['pixel', 'clip_vit_b16', 'clip_vit_l14'], type=str, help='define target type for loss')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--per_frame_crop', action='store_true',
                        help='Apply different crop to each frame (default: False)')
    parser.add_argument('--scales', type=float, nargs='+', default=[1, .875, .75, .66],)

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--wandb_project', type=str, default="pretrain")
    parser.add_argument('--wandb_entity', type=str, default="trackmae")
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)

    parser.add_argument('--loss_lambda', default=1.0, type=float)

    parser.add_argument('--cotracker_norm', default='patch', type=str,
                        help='normalization method for cotracker: none, patch')

    parser.add_argument('--traj_only', action='store_true',
                        help='train with only trajectory loss')

    parser.add_argument('--compile_model', action='store_true',
                        help='Compile the VideoMAE model with torch.compile.')
    parser.add_argument('--compile_cotracker', action='store_true',
                        help='Compile the CoTracker model with torch.compile.')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")

    if args.target_type=='pixel':
        dec_dim = 1536

    if 'clip' in args.target_type:
        from clip import build_clip_feature_extractor, get_clip_feature_dim
        feature_extraction_model = build_clip_feature_extractor(
            model_name=args.target_type,
        )
        dec_dim = get_clip_feature_dim(model_name=args.target_type)
    else:
        feature_extraction_model = None

    model = create_model(
        args.model,
        pretrained=False,
        img_size=args.input_size,
        all_frames=args.num_frames,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        tracker_decoder_num_classes=8,
        tracker_decoder_depth=4,
        use_checkpoint=args.use_checkpoint,
        use_cotracker=True,
        decoder_num_classes=dec_dim,
    )

    return model, feature_extraction_model


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


def main(args, wandb_run=None):
    print(args)

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model, feature_extraction_model = get_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if feature_extraction_model is not None:
        feature_extraction_model.to(device)

    print("Model = %s" % str(model))
    print('number of params: {} M'.format(n_parameters / 1e6))

    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    print(f"Window Size = {args.window_size}")
    args.patch_size = patch_size
    if args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    if args.upsample:
        print("Grid size: 14")
        cotracker_model = CoTrackerFF(grid_size=14)
    else:
        print("Grid size: 28")
        cotracker_model = CoTrackerFF(grid_size=28)
    cotracker_model.to(device)
    if args.compile_cotracker:
        cotracker_model = torch.compile(cotracker_model, mode="reduce-overhead")
    cotracker_model.eval() 

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True,
    )

    prefetcher = CUDAPrefetcher(data_loader_train, device)

    torch.set_float32_matmul_precision('high')
    if args.distributed:
        if args.traj_only:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    model_for_checkpoint = unwrap_compiled_model(model_without_ddp)

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(
        args, model_for_checkpoint)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_for_checkpoint, optimizer=optimizer, loss_scaler=loss_scaler)
    torch.cuda.empty_cache()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        prefetcher.reset()

        train_stats = train_one_epoch(
            model, prefetcher,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
            args=args,
            cotracker_model=cotracker_model,
            ratio_motion=args.ratio_motion,
            mask_type=args.mask_type_cotracker,
            wandb_run=wandb_run,
            feature_extraction_model=feature_extraction_model,
            target_type=args.target_type,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_for_checkpoint, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            utils.save_model(
                    args=args, model=model, model_without_ddp=model_for_checkpoint, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, last_epoch=True)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters}

            if utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize distributed mode
    utils.init_distributed_mode(opts)
    print(f"Rank {opts.rank} reached barrier, distributed={opts.distributed}")
    print(f"Rank {opts.rank} using GPU {torch.cuda.current_device()}")
    torch.distributed.barrier()
    print(f"Rank {opts.rank} passed barrier")
    utils.setup_for_distributed(opts.rank == 0)
    print("DDP initialized successfully")

    # Initialize wandb
    if utils.is_main_process():
        if opts.wandb_name:  # Only log if wandb_name is provided
            wandb.login(key=opts.wandb_key)
            wandb_run = wandb.init(project=opts.wandb_project, entity=opts.wandb_entity, config=vars(opts), tags=["pretraining"], name=opts.wandb_name, resume="allow", id=opts.wandb_name)
        else:
            wandb_run = None
    else:
        wandb_run = None

    main(opts, wandb_run)
    if opts.distributed:
        torch.distributed.destroy_process_group()
