from paead_info import *
from paead_utils import *
import string
import re
import warnings


class Field():
	def __init__(self, start_idx, end_idx, text, field_text, field_name, attribute=None):
		original_field_text = (field_text + '.')[:-1]
		transformations = [(lambda x: ' '.join(x.split())), (lambda x: x.title()), (lambda x: x.upper()), (lambda x: x.lower())]

		def fix_indexes(field_text, transform):
			if transform:
				new_field = transform(field_text)
				match = re.search(r'\b' + f'({re.escape(new_field)})', text)
			else:
				new_field = field_text
				match = re.search(f'({re.escape(new_field)})', text)
			start_idx = match.start() if match else -1
			end_idx = match.end() if match else -1
			if start_idx >= 0:
				return new_field, start_idx, end_idx
			return None

		while start_idx < 0 and transformations:
			transformation = transformations.pop(0)
			res = fix_indexes(field_text, transformation)
			if res:
				field_text, start_idx, end_idx = res
		if start_idx < 0:
			res = fix_indexes(field_text, None)
			if res:
				field_text, start_idx, end_idx = res

		assert text[start_idx:end_idx] == field_text or '#' in field_text, f"Field Error: text span ({field_text}) and indexes ({text[start_idx:end_idx]}) don't match for field {field_name}, {text}"  #or '<empty>' == field_text
		if start_idx > 0 and (not text[start_idx-1] in (string.punctuation + ' ”¦“‚‘ô') and field_text not in 'LGBT' and field_text not in 'lgbt'):
			print(text[start_idx-1:], '---', field_text)
			exit()
		self.start_idx = start_idx
		self.end_idx = end_idx
		self.field_text = original_field_text
		self.field_tokenized_text = tokenize(field_text)
		self.field_name = FIELDS_TO_NAMES[field_name] if field_name in FIELDS_TO_NAMES else field_name
		self.subfields = {}
		self.attribute = attribute if USE_ATTRIBUTES else None
		if not USE_ATTRIBUTES and field_name == 'stance':  #Stance becomes two different slots 
			self.field_name = 'positive_stance' if attribute == 'support' else 'negative_stance'


class Instance():
	def __init__(self, annotation):
		if isinstance(annotation, list):
			instance_json, annotation_json = annotation
			self.qID = annotation_json['qid']
			self.copyID = annotation_json['copyid']
			self.fullID = f'{self.qID}_{self.copyID}'
			self.rule = annotation_json['rule'] if not (annotation_json['rule'] == 'animosity') else 'derogation'
			self.text = instance_json['text']
			self.tokenized_text = tokenize(self.text)
			assert len(annotation_json) == 3, f'Annotation should contain no more than 3 fields, {len(annotation_json)} found.'
		else:
			self.qID = annotation['qid']
			self.copyID = annotation['copyid']
			self.opinionID = annotation['opinionid']
			self.fullID = f'{self.qID}_{self.copyID}_{self.opinionID}'
			self.rule = annotation['rule'] if not (annotation['rule'] == 'animosity') else 'derogation'
			self.text = annotation['text']
			self.tokenized_text = tokenize(self.text)

			self.subfields = {}
			if 'targets' in annotation:
				target_text = annotation['targets'][0]
				match = re.search(r'\b' + f'({re.escape(target_text)})', self.text)
				start_idx = match.start() if match else -1
				end_idx = match.end() if match else -1
				# start_idx = self.text.find(target_text)
				# end_idx = start_idx + len(target_text)
				self.subfields['target'] = Field(start_idx, end_idx, self.text, target_text, 'target')
				if 'protected_characteristics' in annotation:
					characteristics = []
					for pc in annotation['protected_characteristics']:
						pc_attr = list(pc.keys())[0]
						pc_match = re.search(r'\b' + f'({re.escape(pc[pc_attr])})', self.text)
						pc_start_idx = pc_match.start() if pc_match else -1
						pc_end_idx = pc_match.end() if pc_match else -1
						characteristics.append(Field(pc_start_idx, pc_end_idx, self.text, pc[pc_attr], 'protected_characteristic', attribute=pc_attr))
					self.subfields['protected_characteristics'] = characteristics

			if 'stance' in annotation:
				stance_attr = list(annotation['stance'].keys())[0]
				if not stance_attr == 'support':
					stance_span = annotation['stance'][stance_attr]
					field_text = ' '.join(stance_span.split())  # Remove multiple spaces
					# start_idx = self.text.find(field_text)
					# end_idx = start_idx + len(field_text)
					match = re.search(r'\b' + f'({re.escape(field_text)})', self.text)
					start_idx = match.start() if match else -1
					end_idx = match.end() if match else -1
					if field_text == '<empty>': field_text = ''
					stance_field = Field(start_idx, end_idx, self.text, field_text, 'stance', attribute=stance_attr)
					self.subfields[stance_field.field_name] = stance_field

			for field_name in RULES_TO_FIELDS[annotation['rule']]:
				key = field_name if not field_name[0] == '?' else field_name[1:]
				assert key in annotation or field_name[0] == '?' or self.rule == 'nothate', f'Required field {key} missing'
				if key in annotation:
					value = annotation[key]
					if isinstance(value, dict):
						attribute = list(value.keys())[0]
						field_text = value[attribute]
					else:
						field_text = ' '.join(annotation[key].split())  # Remove multiple spaces
						attribute = None
					match = re.search(r'\b' + f'({re.escape(field_text)})', self.text)
					start_idx = match.start() if match else -1
					end_idx = match.end() if match else -1
					self.subfields[key] = Field(start_idx, end_idx, self.text, field_text, key.split('_')[0], attribute=attribute)


class InstanceByTask(Instance):
	def __init__(self, annotation, task_name, predicted_chunks=None):
		assert task_name in AVAILABLE_TASKS, f'{task_name} is not a valid task'
		super().__init__(annotation)
		self.label, self.tokenized_label = self.__set_label__(task_name)
		if predicted_chunks:
			self.automatic_chunks = [c for c in predicted_chunks if c not in string.punctuation]

	def __set_label__(self, task_name):
		if task_name == 'binary_classification':
			rule = self.rule if not (self.rule == 'animosity') else 'derogation'
			label = 1 if rule in HATEFUL_RULES else 0
			tokenized_label = None
		elif task_name == 'classification':
			rule = self.rule if not (self.rule == 'animosity') else 'derogation'
			label = RULES.index(rule)
			tokenized_label = None
		elif task_name == 'intent_and_slot_filling':

			def make_slot(field, isTarget=False):
				if field.attribute is not None:
					attribute_slot = f'[ATTR:{title(field.attribute)},{field.field_text}]'
					attribute_tokenized_slot = ['[', f'ATTR:{title(field.attribute)}', ','] + field.field_tokenized_text + [']']
				else:
					attribute_slot = f'{field.field_text}'
					attribute_tokenized_slot = field.field_tokenized_text
				
				slot = f'[SL:{title(field.field_name)},{attribute_slot}]'
				tokenized_slot = ['[', f'SL:{title(field.field_name)}', ','] + attribute_tokenized_slot + [']']
				return slot, tokenized_slot

			intent = title(self.rule) if not self.rule == 'nothate' else 'NotHateful'
			label = f'[IN:{intent}'
			tokenized_label = ['[', f'IN:{intent}']
			for key, field in self.subfields.items():
				if isinstance(field, list):
					assert key == 'protected_characteristics', f'Multiple fields with key {key} found.'
					for ff in field:
						sublabel, subtokenized_label = make_slot(ff)
						label += ',' + sublabel
						tokenized_label += [','] + subtokenized_label
				else:
					sublabel, subtokenized_label = make_slot(field)
					label += ',' + sublabel
					tokenized_label += [','] + subtokenized_label
			label += ']'
			tokenized_label += [']']
		return label, tokenized_label


class PlaceholderInstance(Instance):
	def __init__(self, tokenized_label):
		self.tokenized_label = tokenized_label
