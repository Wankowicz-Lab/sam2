"""
Analyze different types of features of mdtraj trajectories.
"""

import numpy as np
import mdtraj
from sam.data.topology import slice_ca_traj


def calc_rmsf(
        traj: mdtraj.Trajectory,
        ref_traj: mdtraj.Trajectory = None,
        ref_index: int = 0,
        ref_coords: str = "mean",
        use_mdtraj: bool = False,
        aggr: str = "mean"
    ) -> np.ndarray:
    traj_c = mdtraj.Trajectory(traj.xyz, topology=traj.topology)
    if ref_traj is None:
        ref_traj = traj_c
    if not use_mdtraj:
        traj_c.superpose(ref_traj, ref_index)
        if ref_coords == "mean":
            mean_pos = traj_c.xyz.mean(axis=0, keepdims=True)
        elif ref_coords == "ref":
            mean_pos = ref_traj.xyz[ref_index:ref_index+1]
        else:
            raise KeyError(ref_coords)
        mean_sd = np.sum(np.square(traj_c.xyz - mean_pos), axis=2)
        if aggr == "mean":
            rmsf = np.sqrt(mean_sd.mean(axis=0))
        elif aggr is None:
            rmsf = np.sqrt(mean_sd)
        else:
            KeyError(aggr)
        return rmsf
    else:
        if ref_coords != "mean":
            raise ValueError()
        ref_traj = mdtraj.Trajectory(ref_traj.xyz, topology=traj.topology)
        rmsf = mdtraj.rmsf(traj_c, ref_traj, ref_index)
        return rmsf


def get_centroid(xyz):
    xyz_c = xyz.mean(axis=0, keepdims=True)
    return xyz_c

# def d_0(n_res):
#     return 1.24*(n_res - 15)**(1/3) - 1.8

# def calc_tm_scores(ca_traj, ca_ref_traj=None):
#     tm_traj = mdtraj.Trajectory(xyz=ca_traj.xyz, topology=ca_traj.topology)
#     # tm_traj.center_coordinates()
#     if ca_ref_traj is None:
#         tm_traj.superpose(tm_traj, 0)
#     else:
#         tm_traj.superpose(ca_ref_traj, 0)
#     xyz = tm_traj.xyz * 10.0
#     if ca_ref_traj is None:
#         cen = get_centroid(xyz)
#     else:
#         cen = ca_ref_traj.xyz[:1,:,:]*10.0
#     sq_dev = np.sum(np.square(cen - xyz), axis=-1)
#     rsq_dev = np.sqrt(sq_dev)
#     n_res = xyz.shape[1]
#     tm_score = (1 / n_res) * np.sum(1 / (1 + (rsq_dev/d_0(n_res))**2), axis=1)
#     rmsd = np.sqrt(np.sum(sq_dev/n_res, axis=1))
#     return tm_score, rmsd*0.1


from sam.data.sequences import ofo_restype_name_to_atom14_names

std_atoms = set()
for k in ofo_restype_name_to_atom14_names:
    for a in ofo_restype_name_to_atom14_names[k]:
        if a:
            std_atoms.add(a)


def calc_q_values(
        traj,
        ref_traj,
        beta=50.0,
        lambda_=1.2,
        delta=0.0,
        threshold=None
    ):

    if len(ref_traj) != 1:
        raise NotImplementedError()
    
    dist = []
    top_atoms_dict = [[a for a in r.atoms if a.name in std_atoms] for r in ref_traj.topology.residues]
    
    top_ids = []
    traj_ids = []
    top_residues = list(ref_traj.topology.residues)  #
    top_atoms = list(ref_traj.topology.atoms)        #
    traj_residues = list(traj.topology.residues)
    traj_atoms = list(traj.topology.atoms)           #
    for i, r_i in enumerate(ref_traj.topology.residues):
        for j, r_j in enumerate(ref_traj.topology.residues):
            if r_j.index - r_i.index > 3:
                a_k_ids = [a_k.index for a_k in top_atoms_dict[i]]
                a_k_atoms = top_atoms_dict[i]
                a_l_ids = [a_l.index for a_l in top_atoms_dict[j]]
                a_l_atoms = top_atoms_dict[j]
                dist_ij = np.sqrt(
                    np.sum(
                        np.square(
                            ref_traj.xyz[:,a_k_ids,None,:] - ref_traj.xyz[:,None,a_l_ids,:]
                        ),
                    axis=-1
                    )
                )[0]
                a_k_pos, a_l_pos = np.unravel_index(dist_ij.argmax(), dist_ij.shape)
                if threshold is not None:
                    if dist_ij[a_k_pos, a_l_pos] > threshold:
                        continue
                top_ids.append((a_k_ids[a_k_pos], a_l_ids[a_l_pos]))
                traj_ids.append(
                    (traj_residues[i].atom(a_k_atoms[a_k_pos].name).index,
                     traj_residues[j].atom(a_l_atoms[a_l_pos].name).index)
                )
                """
                dist_ij = []
                for a_k in top_atoms_dict[i]:
                    for a_l in top_atoms_dict[j]:
                        d_kl = np.linalg.norm(ref_traj.xyz[:,a_k.index] - ref_traj.xyz[:,a_l.index])
                        dist_ij.append((d_kl, a_k, a_l))
                dist_ij = sorted(dist_ij, key=lambda t: t[0])[0]
                # dist.append(dist_ij + (i, j))
                top_ids.append((dist_ij[1].index, dist_ij[2].index))
                traj_ids.append(
                    (traj_residues[i].atom(dist_ij[1].name).index, traj_residues[j].atom(dist_ij[2].name).index)
                )
                """
                # print("---")
                # print(top_atoms[top_ids[-1][0]], top_atoms[top_ids[-1][1]])
                # print(traj_atoms[traj_ids[-1][0]], traj_atoms[traj_ids[-1][1]])
    if not traj_ids:
        raise ValueError()
    ref_dist = mdtraj.compute_distances(ref_traj, top_ids)
    traj_dist = mdtraj.compute_distances(traj, traj_ids)
    n_contacts = ref_dist.shape[1]
    q_x = np.sum(1/(1+np.exp(beta*(traj_dist - lambda_*(ref_dist + delta)))), axis=1)/n_contacts
    # score = q_x.mean
    return q_x



def tmscore(hat_traj, ref_traj, ref_idx=0, prealigned=True, get_rmsd=False):
    ref_traj = ref_traj[ref_idx:ref_idx+1]
    if not prealigned:
        raise NotImplementedError()
    ref_xyz = ref_traj.xyz * 10.0
    hat_xyz = hat_traj.xyz * 10.0
    sq_dev = np.sum(np.square(ref_xyz - hat_xyz), axis=-1)
    rsq_dev = np.sqrt(sq_dev)
    n_res = ref_xyz.shape[1]
    tm_score = (1 / n_res) * np.sum(1 / (1 + (rsq_dev/d_0(n_res))**2), axis=1)
    if not get_rmsd:
        return tm_score
    else:
        rmsd = np.sqrt(sq_dev.sum(axis=1)/n_res)
        # rmsd_md = mdtraj.rmsd(hat_traj, ref_traj)
        return tm_score, rmsd*0.1

def d_0(n_res):
    return 1.24*(n_res - 15)**(1/3) - 1.8
    
def calc_tmscore_to_init(traj, init_traj, is_ca=False, get_rmsd=False):
    if not is_ca:
        init_traj = slice_ca_traj(init_traj)
        traj = slice_ca_traj(traj)
    traj.superpose(init_traj)
    scores = tmscore(
        traj, init_traj, ref_idx=0, prealigned=True, get_rmsd=get_rmsd
    )
    return scores

def calc_se_values(traj, ref_traj):
    if len(ref_traj) > 1:
        raise ValueError()
    ref_dssp = mdtraj.compute_dssp(ref_traj)
    ref_dssp = np.isin(ref_dssp, ['H', 'E'])
    ref_se = ref_dssp.mean(axis=1)
    
    traj_dssp = mdtraj.compute_dssp(traj)
    traj_dssp = np.isin(traj_dssp, ['H', 'E'])
    traj_se = traj_dssp.mean(axis=1)
    score = traj_se / ref_se
    return score

def calc_sep_values(traj, ref_traj):
    if len(ref_traj) > 1:
        raise ValueError()
    ref_dssp = mdtraj.compute_dssp(ref_traj)
    traj_dssp = mdtraj.compute_dssp(traj)
    match = ref_dssp == traj_dssp
    match = match.astype(int)
    score = match.mean(axis=1)
    he_mask = np.isin(ref_dssp, ['H', 'E']).astype(int)
    match_mask = match * he_mask
    # print(score.size, score.mean(axis=1).mean(axis=0), he_mask.sum())
    score_mask = match_mask.sum(axis=1)/he_mask.sum()
    return score, score_mask

def _calc_distances(xyz, eps=1e-9):
    d = np.sqrt(
        np.sum(np.square(xyz[:,:,None,:] - xyz[:,None,:,:]), axis=3)+eps
    )
    return d

def calc_lddt_to_ref(
        traj,
        ref_traj,
        ref_idx=0,
        is_ca=True,
        thresholds=[0.05, 0.1, 0.2, 0.4]
    ):
    if not is_ca:
        raise NotImplementedError()
    
    hat_xyz = traj.xyz
    ref_xyz = ref_traj.xyz[ref_idx:ref_idx+1]
    L = ref_xyz.shape[1]
    triu_ids = np.triu_indices(L, k=1)
    ref_dists = _calc_distances(ref_xyz)  # (B, L, L)
    hat_dists = _calc_distances(hat_xyz)  # (B, L, L)
    ref_dists = ref_dists[:, triu_ids[0], triu_ids[1]]
    hat_dists = hat_dists[:, triu_ids[0], triu_ids[1]]
    # Score lDDT for ref-hat conformations.
    lddt = []
    ref_dists_mask = ref_dists <= 1.5
    ref_dists_mask = ref_dists_mask.astype(int)
    delta_dists = abs(ref_dists - hat_dists)
    lddt = 0
    for t in thresholds:
        preserved_t = delta_dists <= t
        preserved_t = preserved_t.astype(int)*ref_dists_mask
        lddt_t = np.sum(preserved_t, axis=1) / ref_dists_mask.sum(axis=1)
        lddt += lddt_t
    lddt = lddt/len(thresholds)
    return lddt
