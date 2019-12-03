# Bagel

A robust and unsupervised KPI anomaly detection algorithm based on conditional variational autoencoder.

## Dependencies

Use Anaconda or Miniconda to manage dependencies:

```
conda env create -f environment.yml
```

## Run

```
cd sample
python main.py
```

KPI data must be stored in csv files in the following format:

```
timestamp,  value,              label
1469376000, 0.8473002740000001, 0
1469376300, -0.036137314,       0
1469376600, 0.074292384,        0
1469376900, 0.074292384,        0
1469377200, -0.036137314,       0
1469377500, 0.184722083,        0
1469377800, -0.036137314,       0
1469378100, 0.184722083,        0
```

`label`: `0` for normal points and `1` for anomaly points.

## Citation

```bibtex
@inproceedings{conf/ipccc/LiCP18,
    author    = {Zeyan Li and
                 Wenxiao Chen and
                 Dan Pei},
    title     = {Robust and Unsupervised {KPI} Anomaly Detection Based on Conditional
                 Variational Autoencoder},
    booktitle = {37th {IEEE} International Performance Computing and Communications
                 Conference, {IPCCC} 2018, Orlando, FL, USA, November 17-19, 2018},
    pages     = {1--9},
    publisher = {{IEEE}},
    year      = {2018},
    url       = {https://doi.org/10.1109/PCCC.2018.8710885},
    doi       = {10.1109/PCCC.2018.8710885}
}
```
