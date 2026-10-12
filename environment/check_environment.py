"""Import dependencies and exercise compiled graph operators on the requested device."""
import argparse
import importlib.metadata
import json
import os
import platform

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with tqdm(total=4, desc='Environment: dependency imports', unit='stage') as progress:
        check_device(args.device, progress)


def check_device(requested_device, progress):
    import torch
    from torch_geometric.nn import GCNConv
    from torch_scatter import scatter_add
    from torch_sparse import SparseTensor
    progress.update(1)
    progress.set_description('Environment: scatter forward/backward')
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside an allocated GPU job and inspect the driver/wheels.")
    device = torch.device(requested_device)
    values = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    groups = torch.tensor([0, 0, 1], device=device)
    result = scatter_add(values, groups, dim=0)
    torch.testing.assert_close(result, torch.tensor([3.0, 3.0], device=device))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    progress.update(1)
    progress.set_description('Environment: sparse operators')
    sparse = SparseTensor(row=groups, col=torch.tensor([0, 1, 1], device=device),
                          value=values.detach(), sparse_sizes=(2, 2))
    torch.testing.assert_close(sparse.matmul(torch.ones((2, 1), device=device)),
                               torch.tensor([[3.0], [3.0]], device=device))
    progress.update(1)
    progress.set_description('Environment: graph convolution and package versions')
    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    graph_conv = GCNConv(1, 1, bias=False).to(device)
    with torch.no_grad():
        graph_conv.lin.weight.fill_(1.0)
    propagated = graph_conv(torch.ones((2, 1), device=device), edge_index)
    torch.testing.assert_close(propagated, torch.ones((2, 1), device=device))
    packages = ("numpy", "pandas", "scikit-learn", "openpyxl", "scipy", "networkx",
                "tqdm", "statsmodels", "matplotlib", "pytest", "torch",
                "torch-geometric", "torch-scatter", "torch-sparse")
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "packages": {p: importlib.metadata.version(p) for p in packages},
                      "torch_cuda": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(),
                      "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "checked_device": str(device), "operator_check": "passed"}, indent=2))
    progress.update(1)


if __name__ == "__main__":
    main()
