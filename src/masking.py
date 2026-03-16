import numpy as np
import torch


class Masker:
    def __init__(self, block_length, mask_fraction, mask_token=-1):
        self.block_length = block_length
        self.mask_fraction = mask_fraction
        self.mask_token = mask_token

    def convert_to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError("Input must be numpy array or torch tensor")

    def mask(self, x):
        x = self.convert_to_tensor(x)

        if x.dim() != 3:
            raise ValueError("Input must be [B, 1, num_snps]")

        B, C, num_snps = x.shape
        if C != 1:
            raise ValueError("Expected input shape [B, 1, num_snps]")

        L = self.block_length
        num_blocks = num_snps // L
        num_masked_blocks = int(num_blocks * self.mask_fraction)

        # choose masked blocks independently for each sample
        rand = torch.rand(B, num_blocks, device=x.device)
        block_indices = torch.topk(rand, num_masked_blocks, dim=1).indices

        block_mask = torch.zeros(B, num_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(1, block_indices, True)

        # expand block mask to SNP mask: [B, num_snps_truncated]
        snp_mask = block_mask.repeat_interleave(L, dim=1)

        # pad if num_snps not divisible by block length
        if snp_mask.shape[1] < num_snps:
            pad = torch.zeros(B, num_snps - snp_mask.shape[1], dtype=torch.bool, device=x.device)
            snp_mask = torch.cat([snp_mask, pad], dim=1)

        # apply mask to channel 0
        masked_x = x.clone().float()
        masked_x[:, 0, :][snp_mask] = self.mask_token

        return masked_x, snp_mask