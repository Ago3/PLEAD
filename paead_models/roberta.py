from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import nn
from torch.nn import functional as f
from paead_info import RM_NUM_CLASSES, RM_MAX_LENGTH, RM_USE_POLICY


class RobertaModel(nn.Module):
	def __init__(self, dataset):
		super(RobertaModel, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=RM_NUM_CLASSES if not RM_USE_POLICY else 2)
		self.model.to(self.device)

	def forward(self, x):
		input_strings = [f'{self.tokenizer.bos_token} {s} {self.tokenizer.sep_token}' for s in x['text']]
		input_ids = self.tokenizer(
			input_strings,
			padding="longest",
			max_length=RM_MAX_LENGTH,
			truncation=True,
			return_tensors="pt"
		)

		input_ids = input_ids.to(self.device)
		x['label'] = x['label'].to(self.device)
		res = self.model(**input_ids, labels=x['label'])
		if self.training:
			return res.loss
		preds = res.logits.argmax(dim=1)
		return preds, x['fullID']
