"""
droid_convmamba.py
==================
ConvMamba-SLAM main entry point.

Since JEGO enrichment is removed, we use DROID's original:
    - FactorGraph (no enrichment needed)
    - DroidFrontend (unchanged)
    - DroidBackend (unchanged)
    - MotionFilter (unchanged — uses net.fnet wrapper)

Only the network changes: DroidNet → ConvMambaSlamNet
"""

import torch
import lietorch
import numpy as np

from convmamba_slam_net import ConvMambaSlamNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.args = args
        self.disable_vis = args.disable_vis
        self.load_weights(args.weights)

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process (original DROID — no JEGO needed)
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process (original DROID — no JEGO needed)
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(
                target=visualization_fn, args=(self.video, None)
            )
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def load_weights(self, weights):
        """Load trained model weights."""

        print(f"Loading weights: {weights}")
        self.net = ConvMambaSlamNet()
        
        ckpt = torch.load(weights)

        # Handle nested checkpoint format (from train_convmamba.py)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
            print(f"  Loaded from checkpoint (step {ckpt.get('total_steps', '?')})")
        else:
            state_dict = ckpt

        # Handle weight/delta head dimension mismatch (DROID compat)
        if "update.weight.2.weight" in state_dict:
            state_dict["update.weight.2.weight"] = \
                state_dict["update.weight.2.weight"][:2]
            state_dict["update.weight.2.bias"] = \
                state_dict["update.weight.2.bias"][:2]
            state_dict["update.delta.2.weight"] = \
                state_dict["update.delta.2.weight"][:2]
            state_dict["update.delta.2.bias"] = \
                state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """Main thread - update map."""
        with torch.no_grad():
            self.filterx.track(tstamp, image, depth, intrinsics)
            self.frontend()

    def terminate(self, stream=None):
        """Terminate visualization, return camera trajectory."""
        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
