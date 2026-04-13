"""
droid_frontend_jamma.py
=======================
DroidFrontend subclass that uses FactorGraphJamma (with JEGO enrichment).

Only change: FactorGraph → FactorGraphJamma with net.jego passed through.
All other frontend logic (initialization, keyframe management, etc.) is inherited.
"""

import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph_jamma import FactorGraphJamma
from droid_frontend import DroidFrontend


class DroidFrontendJamma(DroidFrontend):
    """DroidFrontend with JEGO-enriched FactorGraph."""

    def __init__(self, net, video, args):
        # Initialize parent (this creates self.graph = FactorGraph(...))
        super().__init__(net, video, args)

        # ★ Replace with JEGO-aware FactorGraph ★
        self.graph = FactorGraphJamma(
            video, net.update,
            max_factors=48,
            upsample=args.upsample,
            jego=net.jego,   # JEGOModule from JammaSlamNet
        )
