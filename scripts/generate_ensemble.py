"""
Generate with SAM conformational ensemble for an input PDB file.

Usage:
    python TODO

"""

import os
import sys
import argparse
import time
import numpy as np
import mdtraj
from sam.model import AllAtomSAM
from sam.utils import read_cfg_file, print_msg
from sam.data.topology import get_seq_from_top
from sam.minimizer.runner import Minimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('-c', '--config_fp', type=str, required=True,
        help='YAML or JSON configuration file for a SAM generative model.')
    parser.add_argument('-i', '--init', type=str, required=True,
        help='Input PDB file with the initial structure.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
        help='Output path. File extensions for different file types will be'
            ' automatically added.')
    parser.add_argument('-u', '--out_fmt', type=str, default='dcd',
        choices=['dcd', 'xtc'],
        help='Output format for the file storing xyz coordinates.'
             ' (default: dcd)')
    parser.add_argument('-n', '--n_samples', type=int, default=250,
        help='Number of samples to generate. (default: 250)')
    parser.add_argument('-t', '--n_steps', type=int, default=100,
        help='Number of diffusion steps. (min=1, max=1000) (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
        help='Batch size for sampling. (default: 8)')
    parser.add_argument('-d', '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'], help='PyTorch device. (default: cuda)')
    parser.add_argument('--temperature', type=float,
        help='temperature (optional, only for temperature-based models)')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode, will not print any output.')
    parser.add_argument('--no_minimize', action='store_true',
        help='Do not perform energy minimization.')
    parser.add_argument('--keep_no_min', action='store_true',
        help='If performing energy minimization, save also a trajectory file'
             ' for the non-minimized ensemble.')
    parser.add_argument('--ca', action='store_true',
        help='Save an additional Ca-only trajectory.')
    parser.add_argument('--time', action='store_true',
        help='Save an output file with the wall clock time of sampling.')
    args = parser.parse_args()


    #---------------
    # Check input. -
    #---------------

    timing = {"all": time.time(), "sample": None}

    if not os.path.isfile(args.init):
        raise FileNotFoundError(args.init)

    model_cfg = read_cfg_file(args.config_fp)
    # check_env(model_cfg)

    tem_traj = mdtraj.load(args.init, top=args.init)
    tbm_data = {"xyz": tem_traj.xyz}
    if model_cfg["generative_stack"]["data_type"] == "aa_protein":
        tbm_data["topology"] = tem_traj.topology
    seq = get_seq_from_top(tbm_data["topology"])

    #-----------
    # Run SAM. -
    #-----------

    # Initialize the SAM model.
    if model_cfg["generative_stack"]["data_type"] == "cg_protein":
        raise NotImplementedError()
    elif model_cfg["generative_stack"]["data_type"] == "aa_protein":
        model_cls = AllAtomSAM
    else:
        raise KeyError(model_cfg["generative_stack"]["data_type"])

    model = model_cls(
        config_fp=args.config_fp,
        device=args.device,
        verbose=not args.quiet
    )

    conditions = {}
    if args.temperature is not None:
        conditions["temperature"] = args.temperature
    sample_args = {}
    
    # Generate ensemble.
    timing["sample"] = time.time()
    out = model.sample(
        seq=seq,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size_eps=args.batch_size,
        batch_size_dec=args.batch_size,
        tbm_data=tbm_data,
        return_enc=False,
        sample_args=sample_args,
        conditions=conditions,
        use_cache=True
    )
    timing["sample"] = time.time() - timing["sample"]

    # Save the output data.
    save = model.save(
        out=out,
        out_path=args.out_path,
        out_fmt=args.out_fmt,
        save_ca=args.ca
    )
    # tem_traj.save(f"{args.out_path}.template.pdb")

    #-------------------------------------
    # Energy minimize the conformations. -
    #-------------------------------------

    if args.no_minimize or model_cfg["minimization"]["protocol"] is None:
        pass
    else:
        timing["min"] = time.time()
        min_obj = Minimizer(
            name="sam_ensemble",
            top_fp=save["aa_top"],
            ens_fp=save["aa_traj"],
            protocol=model_cfg["minimization"]["protocol"]
        )
        min_traj = min_obj.run(device=args.device, verbose=not args.quiet)
        if args.keep_no_min:
            min_out_str = ".min"
        else:
            min_out_str = ""
        min_traj_path = f"{args.out_path}{min_out_str}.traj.{args.out_fmt}"
        print_msg(
            f"- Saving a trajectory file to: {min_traj_path}",
            verbose=not args.quiet,
            tag="minimization"
        )
        min_traj.save(min_traj_path)
        min_top_path = f"{args.out_path}{min_out_str}.top.pdb"
        print_msg(
            f"- Saving a topology PDB file to: {min_top_path}",
            verbose=not args.quiet,
            tag="minimization"
        )
        min_traj[0].save(min_top_path)
        timing["min"] = time.time() - timing["min"]

    #------------
    # Complete. -
    #------------

    timing["all"] = time.time() - timing["all"]
    if args.time:
        with open(f"{args.out_path}.time.txt", "w") as o_fh:
            for stage in timing:
                o_fh.write(f"{stage}: {timing[stage]}\n")