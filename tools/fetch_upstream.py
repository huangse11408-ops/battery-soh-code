# -*- coding: utf-8 -*-
'''Clone the pinned upstream backbones into vendor/.

Repositories and commits are the ones used for the reported numbers. Nothing is
copied into version control; rerun this after a fresh checkout.
'''

import subprocess
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor"

SOURCES = [
    ("dlinear", "https://github.com/cure-lab/LTSF-Linear.git", "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"),
    ("mixlinear", "https://github.com/aitianma/MixLinear.git", "main"),
    ("patchtst", "https://github.com/yuqinie98/PatchTST.git", "204c21efe0b39603ad6e2ca640ef5896646ab1a9"),
    ("itransformer", "https://github.com/thuml/iTransformer.git", "main"),
    ("moderntcn", "https://github.com/luodhhh/ModernTCN.git", "main"),
    ("timemixer", "https://github.com/kwuking/TimeMixer.git", "main"),
    ("timemixerpp", "https://github.com/WenjieDu/PyPOTS.git", "main"),
]


def run(command, cwd):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main():
    VENDOR.mkdir(parents=True, exist_ok=True)
    for name, repository, commit in SOURCES:
        target = VENDOR / name
        if target.exists():
            print("present", name)
            continue
        run(["git", "clone", "--depth", "1", repository, name], VENDOR)
        if commit != "main":
            run(["git", "fetch", "--depth", "1", "origin", commit], target)
            run(["git", "checkout", commit], target)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
