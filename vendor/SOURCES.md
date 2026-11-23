# Pinned upstream sources

Pinned revisions used for the reported baseline numbers. Run
`python tools/fetch_upstream.py` to clone them into this directory; the code is
never committed here.

| Name | Repository | Revision |
| --- | --- | --- |
| DLinear | https://github.com/cure-lab/LTSF-Linear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` |
| MixLinear | https://github.com/aitianma/MixLinear | default branch |
| PatchTST | https://github.com/yuqinie98/PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` |
| iTransformer | https://github.com/thuml/iTransformer | default branch |
| ModernTCN | https://github.com/luodhhh/ModernTCN | default branch |
| TimeMixer | https://github.com/kwuking/TimeMixer | default branch |
| TimeMixer++ | https://github.com/WenjieDu/PyPOTS | default branch |

Each project keeps its own license. Adaptations applied here are limited to the
input tokenisation and a scalar regression readout; the backbone mathematics is
unchanged.
