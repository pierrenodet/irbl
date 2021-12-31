# Importance Reweighting for Biquality Learning

This repository provides a reference implementation of the algorithm described in the paper :

[Importance Reweighting for Biquality Learning](https://arxiv.org/abs/2010.09621)

## Overview

Importance Reweighting for Biquality Learning (IRBL) is a meta algorithm that learns to reweight untrusted examples from a small trusted dataset to detect corrupted data and learn efficient classifiers under complex supervision deficiencies.

## Replication

In order to run the experiments and generate results files and figures, run the following lines :

```
git clone https://github.com/pierrenodet/irbl.git
cd irbl
unzip data.zip
python src/main.py
```

Otherwise detailed results are provided for both supervisions deficiencies ([NCAR](results/ncar.csv) and [NNAR](results/nnar.csv)) in the [results directory](results).

## Citation

If you use IRBL in your research, please consider citing us :

```
@inproceedings{nodet2021importance,
  title={Importance reweighting for biquality learning},
  author={Nodet, Pierre and Lemaire, Vincent and Bondu, Alexis and Cornu{\'e}jols, Antoine and Ouorou, Adam},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2021},
  organization={IEEE}
}
```
