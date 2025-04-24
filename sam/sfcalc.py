import numpy as np
import os
import time
import sys
import math
import gemmi
import random

import torch
import torch.nn as nn

import reciprocalspaceship as rs
import SFC_Torch as SFC
from SFC_Torch import PDBParser as sfparser

def linlogcut(x, a=0, b=1000):
    """Function which is linear until a, logarithmic until b and then constant.

    y = x                  x <= a
    y = a + log(x-a+1)   a < x < b
    y = a + log(b-a+1)   b < x

    """
    # cutoff x after b - this should also cutoff infinities
    x = torch.where(x < b, x, b * torch.ones_like(x))
    # log after a
    y = a + torch.where(x < a, x - a, torch.log(x - a + 1.0))
    # make sure everything is finite
    y = torch.where(torch.isfinite(y), y, b * torch.ones_like(y))
    return y

class NLLloss:
    """
    Crystallographic Log Likelihood loss, using (3a) and (3b) from Read, 2016
    """
    def __init__(self, dcp, unit_change=1.0) -> None:
        self.dcp = dcp
        self.n_atoms = dcp.n_atoms
        self.unit_change = unit_change
        self.working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    def get_acentric_NLL(self, sigmaA, Eo, Ec, sigmaE):
        """
        Acentric reflections
        """
        inf_variance = (1 - sigmaA.square().unsqueeze(-1) + 2*sigmaE.square().unsqueeze(0)) # [N_batch, N_hkl]
        bessel_arg = (2 * sigmaA.unsqueeze(-1) * Eo * Ec) / inf_variance # [N_batch, N_hkl]
        exp_bessel = torch.special.i0e(bessel_arg) # [N_batch, N_hkl]
        nll_a = (Eo - sigmaA.unsqueeze(-1) * Ec).square() / inf_variance - torch.log(exp_bessel) - torch.log(2*Eo/inf_variance) # [N_batch, N_hkl]
        return nll_a.sum(-1)
    
    def get_centric_NLL(self, sigmaA, Eo, Ec, sigmaE):
        """
        Centric reflections
        """
        inf_variance = (1 - sigmaA.square().unsqueeze(-1) + sigmaE.square().unsqueeze(0)) # [N_batch, N_hkl] 
        cosh_arg = (sigmaA.unsqueeze(-1) * Eo * Ec) / inf_variance # [N_batch, N_hkl] 
        nll_c = (Eo.square() + sigmaA.square().unsqueeze(-1) * Ec.square()) / (2 * inf_variance) - self.logcosh(cosh_arg) - 0.5*torch.log(2/(math.pi*inf_variance)) # [N_batch, N_hkl]
        return nll_c.sum(-1)
    
    def logcosh(self, x):
        """
        https://stackoverflow.com/questions/57785222/avoiding-overflow-in-logcoshx
        """
        # s always has real part >= 0
        s = torch.sign(x) * x
        p = torch.exp(-2 * s)
        return s + torch.log1p(p) - math.log(2)

    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0, sub_ratio=1.0) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fc_batch = self.dcp.calc_ftotal_batch()
        # Do ensemble average first
        Fc_average = torch.nanmean(Fc_batch, dim=0)
        Ec_average = self.dcp.calc_Ec(Fc_average)
        sigmaAs = self.dcp.get_sigmaA(Ec_average)
        NLL = 0.0
        # Do a subsampling of the working index
        working_set = self.working_set.copy()
        num_true_to_false = int((1.0 - sub_ratio) * working_set.sum())
        true_indices = np.random.choice(np.flatnonzero(working_set), size=num_true_to_false, replace=False)
        working_set[true_indices] = False
        for i in range(self.dcp.n_bins):
            index_i = self.dcp.bins[working_set] == i
            Ec_i = Ec_average[working_set][index_i].abs()
            Eo_i = self.dcp.Eo[working_set][index_i]
            SigEi = self.dcp.SigEo[working_set][index_i]
            Centrici = self.dcp.centric[working_set][index_i]
            NLL_a = self.get_acentric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[~Centrici],
                Ec=Ec_i[~Centrici],
                sigmaE=SigEi[~Centrici]
            )
            NLL_c = self.get_centric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[Centrici],
                Ec=Ec_i[Centrici],
                sigmaE=SigEi[Centrici]
            )
            NLL_i = NLL_a + NLL_c
            NLL = NLL + w_NLL * NLL_i
        NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        return NLLreg

def exp_nll(pred_coords, pdb_path, mtz_path, device):

    pdb_file = os.path.join(pdb_path) 
    mtz_file = os.path.join(mtz_path) 
    
    structure = gemmi.read_pdb(pdb_file)
    structure.remove_waters()
    structure.remove_empty_chains()

    dcp = SFC.SFcalculator(
        sfparser(structure), 
        mtz_file, 
        expcolumns=['FP', 'SIGFP'], 
        set_experiment=True, 
        freeflag='FREE', 
        testset_value=0,
        device = device
        )

    dcp.inspect_data()
    dcp.calc_fprotein()
    dcp.calc_fsolvent()
    dcp.get_scales_adam()

    reflections = torch.tensor(dcp.HKL_array.shape[0], device=device)
    dcp.atom_pos_orth = pred_coords
    
    nll = NLLloss(dcp)
    loss = nll(pred_coords.reshape(1, -1, 3))
    
    return loss, reflections, dcp.r_free