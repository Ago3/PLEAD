from paead_evaluation import IntentSlotEval
from paead_preprocessing import PlaceholderInstance


PREDICTIONS = [
	[],
	['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']'],
	['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Race', ',', 'girls', ']', ']', ']'],
	['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']'],
	['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']']
	]
REFERENCES = [
	[['[', 'IN:NotHateful', ']']],
	[['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'women', ']', ']', ']']],
	[['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']']],
	[['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'women', ']', ']', ']'], ['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']']],
	[['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'girls', ']', ']', ']'], ['[', 'IN:ThreateningSpeech', ',', '[', 'SL:Target', ',', 'they', ']', ',', '[', 'SL:ProtectedCharacteristic', ',', '[', 'ATTR:Gender', ',', 'women', ']', ']', ']']]
	]
EMA_JACCARD_SIMILARITY_SCORES = [
	0,
	0.75,
	0.6667,
	1,
	1
	]


def ema_jaccard_similarity(corpus):
	evalTool = IntentSlotEval(corpus, PREDICTIONS, PREDICTIONS)
	for prediction, references, exp_score in zip(PREDICTIONS, REFERENCES, EMA_JACCARD_SIMILARITY_SCORES):
		instances = [PlaceholderInstance(reference) for reference in references]
		score = evalTool.__ema_jaccard_similarity_test__(instances, prediction, False)
		assert round(score, 4) == exp_score, f'Error in EMA_JACCARD_SIMILARITY: Expected score was {exp_score}, got {round(score, 4)}'


def functionality_tests(corpus):
	# ema_jaccard_similarity(corpus)
	corpus.get_human_score(['f1_cfm_per_token_productions', 'f1_cfm_per_chunk_productions', 'f1_cfm_per_token_only_slots_productions', 'f1_cfm_per_chunk_only_slots_productions', 'f1_cfm_top_level_productions', 'ema_tree'])
