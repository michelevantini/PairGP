# PairGP
PairGP: Gaussian process modeling of longitudinal data from paired multi-condition studies

We propose PairGP, a non-stationary Gaussian process method to compare gene expression timeseries across several conditions that can account for paired longitudinal study designs and can identify groups of conditions that have different gene expression dynamics. We demonstrate the method on both simulated data and previously unpublished RNA-seq time-series with five conditions. The results show the advantage of modeling the pairing effect to better identify groups of conditions with different dynamics.

https://www.biorxiv.org/content/10.1101/2020.08.11.245621v1

## Set-up and installation
To run the PairGP software we suggest to create a Python virtual environment and to install the requirements:
```
python3 -m venv pairgp_env
source pairgp_env/bin/activate
pip install --upgrade pip
pip install -r requirements/base.txt
```

To be able to follow the tutorial in `notebooks/notebook.ipynb`, please also install and run Jupyter:
```
pip install jupyterlab
jupyter lab

```

## Usage
You can find a step-by-step tutorial on how to use the PairGP software and how to prepare your data in the jupyter notebook `notebooks/notebook.ipynb`.

