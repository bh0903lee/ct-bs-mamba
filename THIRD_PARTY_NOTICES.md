# Third-party notices

## Transformer Hawkes Process

The following files are adapted from the Transformer Hawkes Process implementation
(https://github.com/SimiaoZuo/Transformer-Hawkes-Process, commit 9f2e563909c538302733f048131ae8c535ce02ec):

- `ctbs/thp_utils/Constants.py`
- `ctbs/thp_utils/Dataset.py`
- `ctbs/thp_utils/Models.py` (only the padding mask `get_non_pad_mask`)
- `ctbs/thp_utils/Utils.py` (log likelihood, type loss and label-smoothing loss)

Modifications: flat imports; only the parts used by CT-BS-Mamba are kept; the dataset reads
sequences stored as dictionaries; the log likelihood uses the linear-interpolation integral;
label-smoothing targets are cast to int64; comments were edited.
`ctbs/thp_utils/Burst.py` is not derived from this repository.

The upstream license:

```
MIT License

Copyright (c) 2024 Simiao Zuo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Software not included

- The intensity-based baselines (RMTPP, NHP, SAHP, AttNHP) were trained and evaluated with
  EasyTPP (https://github.com/ant-research/EasyTemporalPointProcess, Apache License 2.0). No
  EasyTPP code is included; `results/` contains only their recorded scores.
- MHP and A-MHP scores in `results/mhp_amhp_point_errors.csv` come from separately trained runs;
  no code of these models is included.
- The selective state-space block in `ctbs/mamba_core.py` and `ctbs_nomb/mamba_core.py` is an
  independent pure-PyTorch implementation of the Mamba layer described by Gu and Dao (2023);
  it contains no code from the official `mamba-ssm` package.

## Data

No dataset files are distributed. `data/rebuild_datasets.py` downloads the source releases
(EasyTPP on Hugging Face, the synthetic exponential dataset released with BMT) or reads the
neural Hawkes process Stack Overflow files supplied by the user. These data remain subject to
the terms of their providers. The repository of the synthetic exponential dataset
(https://github.com/bh0903lee/BMT-synthetic-datasets) is distributed under the MIT License; its
license file was added in commit de7d3bd5bc1e54b79295112fb68db29bb50d800f, whose parent is the
pinned data revision 1fbbcad28f4068a8b4d5b13dcbce7adf27d94a8e.
