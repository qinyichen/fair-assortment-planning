# Fair Assortment Planning

This repository contains the implementation of the algorithms and experiments presented in the paper "Fair Assortment Planning". The scripts enable the replication of the results and the execution of synthetic experiments related to the paper.

## Paper Link

For more details on the research and findings, please refer to our paper. [Fair Assortment Planning](<https://arxiv.org/abs/2208.07341>).

## Contents

The repository includes the following Python scripts:

- `approx_alg.py`: The approximate algorithm (1/2-approx. and FPTAS) for maximizing the cost-adjusted revenue function.
- `column_generation.py`: Apply column generation method, which uses approx_alg as separation oracles, to solving FAP.
- `experiment.py`: The main script to run synthetic experiments.
- `primal.py`: solving primal problem as part of column_generation
- `staticMNL.py`: [staticMNL algorithm](<https://github.com/thejat/scalable-data-driven-assortment-planning/tree/master>) used for benchmarking purpose

## Getting Started

To run a synthetic experiment, execute the following command in the terminal:

```
python experiment.py <number>
```

where the number indicates the instance and the fairness level $\delta$ to be adopted

## Citing Our Work

If you utilize this code for your research or project, please acknowledge our paper by citing:

```bibtex
@misc{chen2023fair,
      title={Fair Assortment Planning}, 
      author={Qinyi Chen and Negin Golrezaei and Fransisca Susan},
      year={2023},
      eprint={2208.07341},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
