# PLEAD

Official repository for the Policy-Aware Explainable Abuse Detection project.

## The PLEAD dataset

You can download our dataset [here](https://github.com/Ago3/PLEAD/tree/main/DATA).

The `dataset.json` file contains a dictionary with key 'annotations' and value a list of annotations. This is how individual annotations look like:
```
{
	"text": "the fucking corona response by those in charge of this country is abysmal, but hopefully black people will die because of this",
	"rule": "derogation",
	"targets": ["black people"],
	"protected_characteristics": [{"race": "black"}],
	"derogation_span": "hopefully#will die",
	"opinionid": 0,
	"_id": {"$oid": "620155a9418b41db266261bf"},
	"qid": "acl13434",
	"copyid": 0,
	"annotator_id": {"$oid": "61bab5e9a937e572d5a2c947"},
	"isindirect": "No"
}
```
Annotations can contain different fields depending on the corresponding intent. You can use `RULES_TO_FIELDS` in `paead_info/info.py` to map intents to the corresponding expected fields.

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
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2022}
}
```