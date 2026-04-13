import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "droid_slam"))

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph

from convmamba_slam_net import ConvMambaSlamNet
from logger import Logger
import wandb

def train(args):
    torch.manual_seed(1234)
    torch.cuda.set_device(0)

    # ── Model ──
    print("\n[1/4] Building ConvMambaSlamNet...")
    model = ConvMambaSlamNet().cuda()
    model.train()

    # ── Resume: model weights only ──
    resume_ckpt = None
    total_steps = 0

    if args.ckpt is not None:
        print(f"  Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cuda:0")
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
            resume_ckpt = ckpt
        else:
            model.load_state_dict(ckpt, strict=False)

    # ── Data ──
    print("\n[2/4] Loading dataset...")
    db = dataset_factory(
        args.datasets, datapath=args.datapath,
        n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax
    )
    print(f"  Dataset size: {len(db)} samples")
    train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=2)

    # ── Optimizer ──
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if resume_ckpt is not None and not args.reset_optimizer:
        # 이어서 학습: optimizer state + total_steps 복원
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        total_steps = resume_ckpt['total_steps']
        # initial_lr, lr 강제 복원
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr
        print(f"  Resumed from step {total_steps}")
    else:
        # optimizer만 리셋, step은 체크포인트에서 가져옴
        if resume_ckpt is not None:
            total_steps = resume_ckpt['total_steps']
        else:
            total_steps = 0
                # initial_lr 세팅 추가
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr
        print(f"  Optimizer reset, starting from step {total_steps}")

    # ── Scheduler: total_steps 확정 후 생성 ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=1e-6, last_epoch=total_steps - 1)

    # LR 확인
    print(f"  LR Check: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"  Scheduler LR: {scheduler.get_last_lr()}")

    # ── Logger ──
    logger = Logger(args.name, scheduler, start_step=total_steps)

    # ── wandb ──
    wandb.init(
        project="ConvMamba-slam",
        name="ConvMamba_slam_v2_from45k",
        config=vars(args),
    )

    print(f"\n[4/4] Starting training for {args.steps} steps...")
    print("=" * 60)
    N = args.n_frames
    save_freq = args.save_freq
    should_keep_training = True

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]

            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            if np.random.rand() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i != j and abs(i - j) <= 2]

            Gs.data[:, 0] = Ps.data[:, 0].clone()
            Gs.data[:, 1:] = Ps.data[:, [1]].clone()
            disp0 = torch.ones_like(disps[:, :, 3::8, 3::8])

            r = 0
            while r < args.restart_prob:
                r = np.random.rand()

                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals = model(
                    Gs, images, disp0, intrinsics0, graph,
                    num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:, :, 3::8, 3::8].detach()

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            logger.push(metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics["total_loss"] = loss.item()
            metrics["learning_rate"] = optimizer.param_groups[0]['lr']
            wandb.log(metrics, step=total_steps)

            if total_steps % 100 == 0:
                print(
                    f"  [{total_steps}/{args.steps}] "
                    f"geo={geo_loss.item():.4f} flo={flo_loss.item():.4f} "
                    f"total={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.8f}"
                )

            if total_steps % save_freq == 0:
                PATH = "checkpoints/%s_%06d.pth" % (args.name, total_steps)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'total_steps': total_steps,
                }, PATH)
                print(f"  💾 [{total_steps}/{args.steps}] Saved → {PATH}")

            if total_steps >= args.steps:
                should_keep_training = False
                break

    PATH = "checkpoints/%s_final.pth" % args.name
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'total_steps': total_steps,
    }, PATH)
    print(f"\n✅ Training complete. Final model → {PATH}")
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ConvMamba_slam_v1')
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--reset_optimizer", action='store_true',
                        help="가중치만 가져오고 optimizer는 새로 시작 (LR 폭증)")
    parser.add_argument("--datasets", nargs="+", default=["tartan"])
    parser.add_argument("--datapath", default="datasets/TartanAir")

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000,
                        help="이번 세션 종료 step")
    parser.add_argument('--total_steps', type=int, default=250000,
                        help="전체 학습 목표 step (LR 커브 기준, 항상 250000)")
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)
    parser.add_argument("--save_freq", type=int, default=1000)

    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    print("=" * 60)
    print("ConvMamba-SLAM Training")
    print("=" * 60)
    print(f"  Steps(session): {args.steps}")
    print(f"  Steps(total):   {args.total_steps}")
    print(f"  Reset optimizer: {args.reset_optimizer}")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    train(args)
