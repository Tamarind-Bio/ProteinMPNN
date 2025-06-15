import numpy as np
import torch
import torch.nn.functional as F

from protein_mpnn_utils import ProteinMPNN

try:
    import ablang
except ImportError:
    ablang = None

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_DICT = {a: i for i, a in enumerate(AA_ALPHABET)}


def seq_to_tensor(seq):
    idx = [AA_DICT.get(a, 0) for a in seq]
    return torch.tensor(idx, dtype=torch.long)[None, :]


def design_with_fusion(struct, start_seq, mask_positions, n_designs=100, T=0.1,
                       seed=None):
    if ablang is None:
        raise ImportError("ablang package is required")

    heavy = ablang.pretrained("heavy")
    heavy.freeze()
    light = ablang.pretrained("light")
    light.freeze()

    mpnn = struct["model"]
    device = next(mpnn.parameters()).device
    rng = np.random.default_rng(seed)

    for _ in range(n_designs):
        seq = list(start_seq)
        order = rng.permutation(mask_positions)
        for pos in order:
            S = seq_to_tensor(seq).to(device)
            logit_mpnn = mpnn.single_site_logits(struct, S, pos)[0].cpu().numpy()
            logit_ab = heavy.single_site_logits(seq, pos) + light.single_site_logits(seq, pos)
            fused = logit_mpnn + logit_ab
            probs = F.softmax(torch.tensor(fused) / T, dim=-1).numpy()
            choice = rng.choice(20, p=probs)
            seq[pos] = AA_ALPHABET[choice]
        yield "".join(seq)
