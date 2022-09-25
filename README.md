# PLEAD

Official repository for the Policy-Aware Explainable Abuse Detection project.

## The PLEAD dataset

You can download our dataset [here](https://github.com/Ago3/PLEAD/tree/main/DATA).

## Model

### Setup

```
git clone https://github.com/Ago3/PLEAD.git
cd PLEAD
conda create -n plead python=3.9.7
conda activate plead
pip install -r requirements.txt
```

### Running experiments

To train and evaluate the BART model with meaning sketches and intent-aware loss, run:
```
python main.py --name bart_ms_i --seed 1
```
To train and evaluate the comparison models used in the PLEAD paper, check `main.py` for instructions.

## Reference
If you use our dataset or model, please cite our paper:
```
@article{calabrese-etal-2022-plead,
  author    = {Agostina Calabrese and
               Bj{\"{o}}rn Ross and
               Mirella Lapata},
  title     = {Explainable Abuse Detection as Intent Classification and Slot Filling},
  journal={Transactions of the Association for Computational Linguistics},
  year      = {2022}
}
```