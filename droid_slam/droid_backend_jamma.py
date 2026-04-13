"""
droid_backend_jamma.py
======================
DroidBackend subclass that uses FactorGraphJamma (with JEGO enrichment).

Changes from DroidBackend:
  - __init__: stores net.jego reference
  - __call__: creates FactorGraphJamma instead of FactorGraph
"""

import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph_jamma import FactorGraphJamma
from droid_backend import DroidBackend


class DroidBackendJamma(DroidBackend):
    """DroidBackend with JEGO-enriched FactorGraph."""

    def __init__(self, net, video, args):
        super().__init__(net, video, args)
        self.jego = net.jego  # JEGOModule from JammaSlamNet

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """main update — with JEGO-enriched correlation."""

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        # ★ FactorGraphJamma instead of FactorGraph ★
        graph = FactorGraphJamma(
            self.video, self.update_op,
            corr_impl="alt",
            max_factors=16 * t,
            upsample=self.upsample,
            jego=self.jego,
        )

        graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True


class DroidAsyncBackendJamma:
    """DroidAsyncBackend with JEGO-enriched FactorGraph."""

    def __init__(self, net, video, args, max_age=7):
        self.video = video
        self.update_op = net.update
        self.jego = net.jego
        self.max_age = max_age

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

        self.graph = FactorGraphJamma(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=-1,
            upsample=self.upsample,
            jego=self.jego,
        )

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """main update"""

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        self.graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        self.graph.update_lowmem(steps=steps, use_inactive=True)
        self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.video.dirty[:t] = True
