"""
factor_graph_jamma.py
=====================
FactorGraph subclass that applies JEGO enrichment before CorrBlock creation.

Changes from original FactorGraph:
  - __init__: accepts `jego` module parameter
  - add_factors: applies JEGO to (fmap_i, fmap_j) before CorrBlock (volume mode)
  - update_lowmem: replaces AltCorrBlock with per-batch JEGO + CorrBlock (alt mode)

Key constraint:
  video.fmaps is NEVER overwritten — JEGO is applied on-the-fly per edge pair,
  because the same frame can be enriched differently depending on its partner.
"""

import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops

from cuda_timer import CudaTimer
from functools import partial

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class FactorGraphJamma(FactorGraph):
    """FactorGraph with JEGO enrichment injected before correlation computation."""

    def __init__(self, video, update_op, device="cuda", corr_impl="volume",
                 max_factors=-1, upsample=False, jego=None):
        super().__init__(video, update_op, device, corr_impl, max_factors, upsample)
        self.jego = jego  # JEGOModule instance (or None to fall back to raw features)

    def _apply_jego(self, fmap1, fmap2):
        """
        Apply JEGO enrichment to feature map pairs.

        Args:
            fmap1: [1, E, C, H, W] — source frame features
            fmap2: [1, E, C, H, W] — target frame features
        Returns:
            fmap1_enriched: [1, E, C, H, W]
            fmap2_enriched: [1, E, C, H, W]
        """
        # Squeeze batch dim: [E, C, H, W]  (JEGO expects [B, C, H, W])
        fi = fmap1.squeeze(0)
        fj = fmap2.squeeze(0)

        print(f"[JEGO] _apply_jego called | edges={fi.shape[0]}, feat={fi.shape[1:]}")
        fi_enriched, fj_enriched = self.jego(fi, fj)

        return fi_enriched.unsqueeze(0), fj_enriched.unsqueeze(0)

    # ──────────────────────────────────────────────
    # Override: add_factors  (frontend — volume corr)
    # ──────────────────────────────────────────────

    @autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """add edges to factor graph — with JEGO enrichment before CorrBlock."""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges (name-mangled private method)
        ii, jj = self._FactorGraph__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(0)

            # ★ JEGO enrichment — applied before CorrBlock ★
            if self.jego is not None:
                fmap1, fmap2 = self._apply_jego(fmap1, fmap2)

            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    # ──────────────────────────────────────────────
    # Override: update_lowmem  (backend — alt corr)
    # ──────────────────────────────────────────────
    #
    # Original uses AltCorrBlock(video.fmaps) which computes correlation
    # on-the-fly from raw features.  We replace this with per-batch
    # JEGO enrichment + CorrBlock to match the training distribution.
    #
    # Trade-off: slightly more memory/compute per batch, but correct
    # feature distribution for the GRU.
    # ──────────────────────────────────────────────

    @autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """run update operator on factor graph — with JEGO enrichment."""

        # If no JEGO module, fall back to original AltCorrBlock path
        if self.jego is None:
            return super().update_lowmem(t0, t1, itrs, use_inactive, EP, steps)

        t = self.video.counter.value
        num, rig, ch, ht, wd = self.video.fmaps.shape

        for step in range(steps):
            with CudaTimer("backend", enabled=False):
                with autocast(enabled=False):
                    coords1, mask = self.video.reproject(self.ii, self.jj)
                    motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                    motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

                s = 8
                for i in range(self.ii.min(), self.jj.max() + 1, s):
                    v = (self.ii >= i) & (self.ii < i + s)
                    iis = self.ii[v]
                    jjs = self.jj[v]

                    if v.count_nonzero().item() == 0:
                        continue

                    ht, wd = self.coords0.shape[0:2]

                    with autocast(enabled=True):
                        # ★ JEGO-enriched correlation (replaces AltCorrBlock) ★
                        c = (iis == jjs).long()
                        fmap1 = self.video.fmaps[iis, 0].to(self.device).unsqueeze(0)
                        fmap2 = self.video.fmaps[jjs, c].to(self.device).unsqueeze(0)

                        fmap1, fmap2 = self._apply_jego(fmap1, fmap2)

                        corr1 = CorrBlock(fmap1, fmap2)(coords1[:, v])

                        net, delta, weight, damping, upmask = \
                            self.update_op(self.net[:, v], self.video.inps[None, iis],
                                           corr1, motn[:, v], iis, jjs)

                        if self.upsample:
                            self.video.upsample(torch.unique(iis), upmask)

                    self.net[:, v] = net
                    self.target[:, v] = coords1[:, v] + delta.float()
                    self.weight[:, v] = weight.float()
                    self.damping[torch.unique(iis)] = damping

                damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP

                if use_inactive:
                    ii = torch.cat([self.ii_inac, self.ii], 0)
                    jj = torch.cat([self.jj_inac, self.jj], 0)
                    target = torch.cat([self.target_inac, self.target], 1)
                    weight = torch.cat([self.weight_inac, self.weight], 1)
                else:
                    ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

                damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
                target = target.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()
                weight = weight.view(-1, ht, wd, 2).permute(0, 3, 1, 2).contiguous()

                self.age += 1

                # dense bundle adjustment
                self.video.ba(target, weight, damping, ii, jj, 1, t,
                              itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

                self.video.dirty[:t] = True