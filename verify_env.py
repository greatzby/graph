#!/usr/bin/env python
# verify_env.py
#
# 用法：
#   conda activate gptenv
#   python verify_env.py
#
# 成功时：
#   - logs/env_smoke_report.json 记录各项检查 detail
#   - env_validation.txt 文件只包含单词 "successful"
#
# 失败时：
#   - 退出码非 0
#   - env_validation.txt 不会出现或内容不是 "successful"

import json
import sys
from pathlib import Path

import torch
import torchvision
import torchaudio
import numpy as np
import pandas as pd
import networkx as nx
from packaging import version

REPORT_PATH = Path("logs") / "env_smoke_report.json"
SUCCESS_PATH = Path("env_validation.txt")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report = {"checks": []}


def add_result(name, ok, detail=""):
    report["checks"].append({"name": name, "status": "pass" if ok else "fail", "detail": detail})
    if not ok:
        raise RuntimeError(f"[{name}] {detail}")


def check_versions():
    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    torchaudio_version = torchaudio.__version__
    numpy_version = np.__version__
    pandas_version = pd.__version__
    networkx_version = nx.__version__

    ok = (
        version.parse(torch_version) >= version.parse("2.0.0")
        and version.parse(torchvision_version) >= version.parse("0.15.0")
        and version.parse(torchaudio_version) >= version.parse("2.0.0")
        and version.parse(numpy_version) >= version.parse("1.24.0")
        and version.parse(pandas_version) >= version.parse("1.5.0")
        and version.parse(networkx_version) >= version.parse("2.8.0")
    )
    add_result(
        "version_check",
        ok,
        detail=(
            f"torch={torch_version}, torchvision={torchvision_version}, "
            f"torchaudio={torchaudio_version}, numpy={numpy_version}, "
            f"pandas={pandas_version}, networkx={networkx_version}"
        ),
    )


def check_torch_cpu_ops():
    torch.manual_seed(42)
    device = torch.device("cpu")
    x = torch.randn(8, 8, device=device)
    w = torch.randn(8, 8, device=device)
    y = x @ w  # 矩阵乘法
    z = torch.nn.functional.relu(y)
    add_result("torch_cpu_ops", True, detail=f"tensor_mean={z.mean().item():.6f}")


def check_torchvision_basic():
    dummy = torch.zeros(1, 3, 32, 32)
    model = torchvision.models.resnet18(weights=None)
    out = model(dummy)
    add_result("torchvision_basic", True, detail=f"output_shape={tuple(out.shape)}")


def check_torchaudio_basic():
    waveform = torch.randn(1, 16000)
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    resampled = resampler(waveform)
    add_result("torchaudio_basic", True, detail=f"resampled_shape={tuple(resampled.shape)}")


def check_networkx_ops():
    g = nx.gn_graph(10, seed=0)  # 有向无环图
    topo = list(nx.topological_sort(g))
    add_result("networkx_ops", True, detail=f"topo_first5={topo[:5]}")


def check_numpy_pandas_ops():
    arr = np.linspace(0, 1, num=5)
    df = pd.DataFrame({"value": arr})
    stats = df.describe().loc["mean", "value"]
    add_result("numpy_pandas_ops", True, detail=f"mean={stats}")


def write_report():
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[verify_env] 报告写入 {REPORT_PATH}")
    print(json.dumps(report, indent=2))


def main():
    try:
        check_versions()
        check_torch_cpu_ops()
        check_torchvision_basic()
        check_torchaudio_basic()
        check_networkx_ops()
        check_numpy_pandas_ops()
    finally:
        write_report()

    # 若全部 pass，则写成功标记文件
    if all(item["status"] == "pass" for item in report["checks"]):
        SUCCESS_PATH.write_text("successful\n", encoding="utf-8")
        print(f"[verify_env] 所有检查通过，写入 {SUCCESS_PATH}")
        return 0
    else:
        # 不写 success 文件
        print("[verify_env] 有检查未通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())