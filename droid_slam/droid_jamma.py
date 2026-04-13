"""
droid_jamma.py
==============
JamMa-SLAM main entry point.

Changes from original droid.py:
  - DroidNet → JammaSlamNet
  - DroidFrontend → DroidFrontendJamma (JEGO-enriched FactorGraph)
  - DroidBackend → DroidBackendJamma (JEGO-enriched FactorGraph)

Unchanged:
  - DepthVideo, MotionFilter, PoseTrajectoryFiller — used as-is
  - MotionFilter uses net.fnet (ConvNeXt only, no JEGO) — by design
  - PoseTrajectoryFiller uses original FactorGraph — post-processing step
"""

import torch
import lietorch
import numpy as np

from jamma_slam_net import JammaSlamNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend_jamma import DroidFrontendJamma as DroidFrontend
from droid_backend_jamma import DroidBackendJamma as DroidBackend
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

        # frontend process — ★ JEGO-enriched ★
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process — ★ JEGO-enriched ★
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        use_uncert = not getattr(self.args, 'no_uncertainty', False)
        self.net = JammaSlamNet(use_uncertainty=use_uncert)

        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights)['model'].items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict, strict=True)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
