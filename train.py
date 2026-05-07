import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "droid_slam"))
import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SE3
from geom import losses
from geom.graph_utils import build_frame_graph

# network
from droid_net import DroidNet
from logger import Logger
import wandb

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl', init_method='env://',     
    	world_size=args.world_size, rank=gpu,
    )

    torch.manual_seed(1234)
    torch.cuda.set_device(gpu)

def move_optimizer_to_device(optimizer, device):
    """Adam state tensors must live on the same device as the params."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def assert_lr_sanity(optimizer, scheduler, total_steps, args, gpu):
    """Catch the LR=0 trap before training starts."""
    current_lr = optimizer.param_groups[0]['lr']
    sched_total = getattr(scheduler, 'total_steps', None)

    if gpu == 0:
        print(f"  [LR Sanity] step={total_steps}, lr={current_lr:.6e}, "
              f"scheduler.total_steps={sched_total}")

    if sched_total is not None and sched_total != args.total_steps:
        raise RuntimeError(
            f"Scheduler total_steps ({sched_total}) != args.total_steps "
            f"({args.total_steps}). LR curve would be wrong. "
            f"Did you change --total_steps between runs? Keep it FIXED at 250000."
        )

    if current_lr < 1e-9 and total_steps < args.total_steps - 100:
        raise RuntimeError(
            f"LR is {current_lr:.2e} at step {total_steps}/{args.total_steps}. "
            f"This is the OneCycleLR exhaustion trap (LR curve already finished). "
            f"Likely cause: previous run used --total_steps too small. "
            f"Fix: delete the checkpoint or use --reset_optimizer."
        )

    if total_steps >= args.total_steps:
        raise RuntimeError(
            f"total_steps ({total_steps}) >= args.total_steps ({args.total_steps}). "
            f"Training already complete. Nothing to do."
        )


def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    is_main = (gpu == 0)
    rng = np.random.default_rng(12345+gpu)
    device = f'cuda:{gpu}'

    if is_main:
        print("\n[1/4] Building Original DroidNet...")
    model = DroidNet().cuda()
    model.train()
    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.total_steps,
        pct_start=0.01, cycle_momentum=False, last_epoch=-1,
    )

    total_steps = 0
    wandb_run_id = None

    if args.ckpt is not None and os.path.isfile(args.ckpt):
        if is_main:
            print(f"\n  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)

        # Validate LR horizon consistency
        ckpt_total = ckpt.get('args', {}).get('total_steps', None)
        if ckpt_total is not None and ckpt_total != args.total_steps:
            raise RuntimeError(
                f"Checkpoint trained with --total_steps={ckpt_total}, "
                f"current run uses --total_steps={args.total_steps}. "
                f"LR curve would be inconsistent. Match the value or use --reset_optimizer."
            )

        # Model
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        state_dict = OrderedDict(
            [(k.replace('module.', ''), v) for (k, v) in state_dict.items()]
        )
        missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
        if is_main:
            if missing:
                print(f"  ⚠ Missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
            if not missing and not unexpected:
                print(f"  ✓ Model state loaded cleanly")

        if isinstance(ckpt, dict):
            if not args.reset_optimizer:
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    move_optimizer_to_device(optimizer, device)
                if 'scheduler' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler'])
                if 'total_steps' in ckpt:
                    total_steps = ckpt['total_steps']
                if 'wandb_run_id' in ckpt:
                    wandb_run_id = ckpt['wandb_run_id']
                if is_main:
                    print(f"  Resumed: step={total_steps}, "
                          f"lr={optimizer.param_groups[0]['lr']:.6e}, "
                          f"wandb_run_id={wandb_run_id}")
            else:
                # Reset everything except model. CRITICAL: total_steps=0 too.
                total_steps = 0
                if is_main:
                    print(f"  Optimizer/scheduler/step reset to 0")
    else:
        if is_main and args.ckpt is not None:
            print(f"  WARNING: --ckpt {args.ckpt} not found, starting fresh.")

    # Sync step across ranks
    ts_tensor = torch.tensor([total_steps], device=device)
    dist.broadcast(ts_tensor, src=0)
    total_steps = int(ts_tensor.item())

    # Hard sanity check before starting
    assert_lr_sanity(optimizer, scheduler, total_steps, args, gpu)

    if is_main:
        print("\n[2/4] Loading dataset...")
    db = dataset_factory(
        args.datasets, datapath=args.datapath,
        n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax,
    )
    if is_main:
        print(f"  Dataset size: {len(db)} samples")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu,
    )
    train_loader = DataLoader(
        db, batch_size=args.batch, sampler=train_sampler, num_workers=2,
    )

    logger = None
    if is_main:
        print("\n[3/4] Setting up logger & wandb...")
        logger = Logger(args.name, scheduler, start_step=total_steps)

        if wandb_run_id is not None:
            wandb.init(
                project="droid-slam-baseline", id=wandb_run_id,
                resume="must", config=vars(args),
            )
            print(f"  Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(
                project="droid-slam-baseline", name=args.name,
                config=vars(args),
            )
            wandb_run_id = wandb.run.id
            print(f"  Started new wandb run: {wandb_run_id}")
    
    if is_main:
        print(f"\n[4/4] Training: {total_steps} → {args.steps} (LR horizon: {args.total_steps})")
        print("=" * 60)

    N = args.n_frames
    save_freq = args.save_freq
    should_keep_training = True
    epoch = 0

    while should_keep_training:
        train_sampler.set_epoch(epoch)
        epoch += 1

        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to(device) for x in item]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()
                
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals = model(Gs, images, disp0, intrinsics0, 
                    graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_steps += 1

            if is_main:
                metrics["total_loss"] = loss.item()
                metrics["learning_rate"] = optimizer.param_groups[0]['lr']
                logger.push(metrics)
                wandb.log(metrics, step=total_steps)

                if total_steps % 100 == 0:
                    print(f"  [{total_steps}/{args.steps}] "
                          f"geo={geo_loss.item():.4f} flo={flo_loss.item():.4f} "
                          f"total={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.8f}")

                if total_steps % save_freq == 0:
                    save_checkpoint(model, optimizer, scheduler, total_steps,
                                    wandb_run_id, args, tag=f"{total_steps:06d}")

            if total_steps >= args.steps:
                should_keep_training = False
                break
    if is_main:
        save_checkpoint(model, optimizer, scheduler, total_steps,
                        wandb_run_id, args, tag="final")
        print(f"\n✅ Session complete at step {total_steps}/{args.total_steps}")
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()
     
def save_checkpoint(model, optimizer, scheduler, total_steps,
                    wandb_run_id, args, tag):
    os.makedirs("checkpoints", exist_ok=True)
    PATH = f"checkpoints/{args.name}_{tag}.pth"
    LATEST = f"checkpoints/{args.name}_latest.pth"

    payload = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_steps': total_steps,
        'wandb_run_id': wandb_run_id,
        'args': vars(args),
    }
    torch.save(payload, PATH)
    tmp = LATEST + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, LATEST)
    print(f"  💾 [{total_steps}] Saved → {PATH}")           

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='droid_slam_baseline')
    parser.add_argument('--ckpt', default=None,
                        help="Resume checkpoint. Use checkpoints/<n>_latest.pth for auto-resume")
    parser.add_argument('--reset_optimizer', action='store_true',
                        help="Load model only; reset optimizer/scheduler/step to 0")
                        
    parser.add_argument('--datasets', nargs='+', default=['tartan'])
    parser.add_argument('--datapath', default='datasets/TartanAir')
    parser.add_argument('--gpus', type=int, default=2,
                        help="GPUs per node. H100=2, A100=3")
    parser.add_argument('--batch', type=int, default=2,
                        help="Per-GPU batch. Effective = gpus * batch")
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000,
                        help="Stop when total_steps reaches this (absolute)")
    parser.add_argument('--total_steps', type=int, default=250000,
                        help="OneCycleLR horizon. NEVER change between resumes.")
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)
    parser.add_argument('--save_freq', type=int, default=10000)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()
    args.world_size = args.gpus
    print("=" * 60)
    print(f"Original DROID-SLAM Baseline (single-node DDP, OneCycleLR)")
    print("=" * 60)
    print(f"  Name:            {args.name}")
    print(f"  GPUs:            {args.gpus}")
    print(f"  Per-GPU batch:   {args.batch}")
    print(f"  Effective batch: {args.gpus * args.batch}")
    print(f"  Steps (stop at): {args.steps}")
    print(f"  LR horizon:      {args.total_steps}  ← MUST stay constant")
    print(f"  LR (max):        {args.lr}")
    print(f"  Save freq:       {args.save_freq}")
    print(f"  Resume from:     {args.ckpt}")
    print(f"  Reset optimizer: {args.reset_optimizer}")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

