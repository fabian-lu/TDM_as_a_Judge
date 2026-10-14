# Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges

Fabian Lukassen<sup>1</sup>, Christoph Weisser<sup>2</sup>, Alexander Silbersdorff<sup>1</sup>  
<sup>1</sup>University of Göttingen, <sup>2</sup>Bielefeld University of Applied Sciences and Arts (HSBI)

[Paper (arXiv)](https://arxiv.org/abs/2610.18122) | [Project page](https://fabian-lu.github.io/TDM_as_a_Judge/)

Typed decision models (TDMs) return a probability for each allowed answer instead of generating text, which makes
them candidates for cheaply replacing LLM judges that compare two answers. We run the bias tests of the CALM
benchmark ([Ye et al., 2025](https://llm-judge-bias.github.io/)), extended by three tests of our own, on eight TDMs
and seven LLM judges of 2024 and 2026.

## What is here

| Folder | For whom |
|---|---|
| [`data/`](data/) | **The benchmark.** The 14,223 prompts of the twelve tests and the answers of all fifteen judges, ready to load. Start with [`data/README.md`](data/README.md). |
| [`code/`](code/) | **Reproducing the paper and testing other judges.** The analysis, the notebooks and the code that sends the prompts to a TDM or an LLM judge. Start with [`code/README.md`](code/README.md). |

To use the benchmark, only `data/` is needed:

```python
import pandas as pd

prompts = pd.read_json("data/prompts.jsonl.gz", lines=True)
answers = pd.read_json("data/answers/tdms/typesafe_jev-1.13__calm.jsonl.gz", lines=True)
```

## Licences

The code is released under the MIT licence (`LICENSE`). The data and our analysis results are released under
CC BY-NC-SA 4.0 (`data/LICENSE`), as required for material adapted from ScienceQA.

## Citation

```bibtex
@misc{lukassen2026cheap,
  title         = {Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges},
  author        = {Lukassen, Fabian and Weisser, Christoph and Silbersdorff, Alexander},
  year          = {2026},
  eprint        = {2610.18122},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.18122}
}
```

Please also cite CALM, on which the benchmark builds:

```bibtex
@inproceedings{ye2025justice,
  title     = {Justice or Prejudice? Quantifying Biases in {LLM}-as-a-Judge},
  author    = {Ye, Jiayi and Wang, Yanbo and Huang, Yue and Chen, Dongping and Zhang, Qihui and Moniz, Nuno and Gao, Tian and Geyer, Werner and Huang, Chao and Chen, Pin-Yu and Chawla, Nitesh V. and Zhang, Xiangliang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```
