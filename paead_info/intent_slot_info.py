from .info import USE_ATTRIBUTES

if USE_ATTRIBUTES:
	ONTOLOGY = [
		'IN:Threatening',
		'IN:Comparison',
		'IN:Hatecrime',
		'IN:Derogation',
		'IN:NotHateful',
		'SL:Target',
		'SL:ProtectedCharacteristic',
		'SL:ThreateningSpeech',
		'SL:EquatedTo',
		'SL:HateEntity',
		'SL:SupportHateCrimes',
		'SL:NegativeOpinion',
		'SL:Stance',
		'ATTR:Race',
		'ATTR:Disability',
		'ATTR:Religion',
		'ATTR:Caste',
		'ATTR:SexualOrientation',
		'ATTR:Gender',
		'ATTR:Disease',
		'ATTR:Immigrant',
		'ATTR:Age',
		'ATTR:Insect',
		'ATTR:IntellectuallyInferiorAnimal',
		'ATTR:PhysicallyInferiorAnimal',
		'ATTR:Filth',
		'ATTR:Bacteria',
		'ATTR:Disease',
		'ATTR:Faeces',
		'ATTR:SexualPredator',
		'ATTR:Subhuman',
		'ATTR:Criminals',
		'ATTR:DenyingExistence',
		'ATTR:Object',
		'ATTR:It',
		'ATTR:FarmEquipment',
		'ATTR:MenialLabourers',
		'ATTR:BestialBehaviour',
		'ATTR:Support',
		'ATTR:Objective',
		'ATTR:Against'
	]
else:
	ONTOLOGY = [
		'IN:Threatening',
		'IN:Comparison',
		'IN:Hatecrime',
		'IN:Derogation',
		'IN:NotHateful',
		'SL:Target',
		'SL:ProtectedCharacteristic',
		'SL:ThreateningSpeech',
		'SL:EquatedTo',
		'SL:HateEntity',
		'SL:SupportHateCrimes',
		'SL:NegativeOpinion',
		'SL:PositiveStance',
		'SL:NegativeStance'
	]

SLOT_TO_ATTRS = {
	'SL:ProtectedCharacteristic': ONTOLOGY[14:23],
	'SL:EquatedTo': ONTOLOGY[23:39],
	'SL:Stance': ONTOLOGY[39:41]
}

TOKENS = ONTOLOGY + ['[', ']', ',']

CAN_COOCCUR = {
	'SL:Target': ['SL:ProtectedCharacteristic', 'SL:ThreateningSpeech', 'SL:EquatedTo', 'SL:NegativeOpinion', 'SL:PositiveStance', 'SL:NegativeStance'],
	'SL:ProtectedCharacteristic': ['SL:ThreateningSpeech', 'SL:EquatedTo', 'SL:NegativeOpinion', 'SL:PositiveStance', 'SL:NegativeStance'],
	'SL:ThreateningSpeech': ['SL:PositiveStance', 'SL:NegativeStance'],
	'SL:EquatedTo': ['SL:PositiveStance', 'SL:NegativeStance'],
	'SL:NegativeOpinion': ['SL:PositiveStance', 'SL:NegativeStance'],
	'SL:HateEntity': ['SL:SupportHateCrimes', 'SL:PositiveStance', 'SL:NegativeStance'],
	'SL:SupportHateCrimes': ['SL:PositiveStance', 'SL:NegativeStance']
}

SLOTS = [token for token in TOKENS if token.startswith('SL:')]

POSITIVELY_AFFECTED_BY = {
	'comparison': [SLOTS.index(s) for s in ['SL:Target', 'SL:ProtectedCharacteristic', 'SL:EquatedTo']],
	'derogation': [SLOTS.index(s) for s in ['SL:Target', 'SL:ProtectedCharacteristic', 'SL:NegativeOpinion']],
	'threatening': [SLOTS.index(s) for s in ['SL:Target', 'SL:ProtectedCharacteristic', 'SL:ThreateningSpeech']],
	'hatecrime': [SLOTS.index(s) for s in ['SL:HateEntity',
		'SL:SupportHateCrimes']],
	'nothate': [SLOTS.index(s) for s in ['SL:NegativeStance']]
}

NEGATIVELY_AFFECTED_BY = {
	'comparison': [SLOTS.index(s) for s in ['SL:NegativeStance']],
	'derogation': [SLOTS.index(s) for s in ['SL:NegativeStance']],
	'threatening': [SLOTS.index(s) for s in ['SL:NegativeStance']],
	'hatecrime': [SLOTS.index(s) for s in ['SL:NegativeStance']],
	'nothate': [SLOTS.index(s) for s in ['SL:Target', 'SL:ProtectedCharacteristic', 'SL:EquatedTo', 'SL:NegativeOpinion', 'SL:ThreateningSpeech', 'SL:HateEntity',
		'SL:SupportHateCrimes']]
}
