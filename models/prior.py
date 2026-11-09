import torch


def aggregate_window(window, agg, device):
    """Aggregate the list of previous predicted-frame encodings, each [N, C] or [B, N, C]-like tensors.

    agg = "avg":          plain mean of the entries
    agg = "weighted_avg": weights 2^i (i = 0 oldest ... newest highest), normalised to sum 1
    Shared by train.py (training and validation) and the demos, so both always aggregate the same way.
    """
    if agg == "avg":
        return torch.mean(torch.stack(window), dim=0).data
    ws = torch.tensor([2 ** i for i in range(len(window))], dtype=torch.float32)
    ws = (ws / ws.sum()).to(device)
    stacked = torch.stack(window, dim=0)
    return torch.sum(stacked * ws.view(len(window), *([1] * (stacked.dim() - 1))), dim=0).data
