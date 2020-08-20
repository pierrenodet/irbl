# Importance Reweighting for Biquality Learning

This repository provides a reference implementation of the algorithm described in the paper :

[Importance Reweighting for Biquality Learning]()

## Overview

Importance Reweighting for Biquality Learning (IRBL) is a meta algorithm that learns to reweight untrusted examples from a small trusted dataset to detect corrupted data and learn efficient classifiers under complex supervision deficiencies.

## Replication

In order to run the experiments and generate results files and figures, run the following lines :

```bash
git clone https://github.com/pierrenodet/irbl.git
cd irbl
unzip data.zip
python src/main.py
```

Otherwise detailed results are provided for both supervisions deficiencies ([NCAR](results/ncar.csv) and [NNAR](results/nnar.csv)) in the [results directory](results).

## Citation

If you use IRBL in your research, please consider citing us :

```

```
