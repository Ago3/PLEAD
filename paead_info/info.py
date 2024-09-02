PC_ATTRIBUTES = [
	'gender',
	'race',  # includes ethnicity and nationality
	'disability',
	'religion',
	'caste',
	'sexual_orientation',
	'disease',
	'immigrant',  # immigration status
	'age'
]


HATEFUL_RULES = ['comparison', 'derogation', 'threatening', 'hatecrime']

NON_HATEFUL_RULES = ['nothate']

RULES = HATEFUL_RULES + NON_HATEFUL_RULES


RULES_TO_FIELDS = {
	'comparison': ['comparison'],
	'threatening': ['threatening_span'],
	'hatecrime': ['entity_span', 'support_span'],
	'derogation': ['derogation_span'],
	'animosity': ['animosity_span'],
	'nothate': ['comparison', 'threatening_span', 'entity_span', 'support_span', 'derogation_span']
}


FIELDS_TO_NAMES = {
	'comparison': 'equated_to',
	'threatening': 'threatening_speech',
	'entity': 'hate_entity',
	'support': 'support_hate_crimes',
	'derogation': 'negative_opinion',
	'animosity': 'negative_opinion'  # Note: Using same name as per derogatory opinion
}

AVAILABLE_TASKS = [
	'aaa',
	'binary_classification',
	'classification',
	'intent_and_slot_filling',
	# 'slots_to_intent',
	'functionality_tests',
	'cad',
        'hatecheck',
]

AVAILABLE_MODELS_BY_TASK = {
	'binary_classification': ['roberta'],
	'classification': ['roberta'],
	'intent_and_slot_filling': ['token_tagging_embeddings','litbart', 'litmsbart', 'litmsbart_with_slot2intent', 'bert_token_tagging'],
	'aaa': ['roberta', 'token_tagging_embeddings', 'litbart', 'litmsbart', 'litmsbart_with_slot2intent', 'bert_token_tagging'],
	'hatecheck': ['roberta', 'token_tagging_embeddings', 'litbart', 'litmsbart', 'litmsbart_with_slot2intent', 'bert_token_tagging'],
	# 'slots_to_intent': ['slots_to_intent'],
	'functionality_tests': [None],
}

DATA_DIR = 'DATA/'

ANNOTATED_DATA = DATA_DIR + 'dataset.json'
EXTENDED_DATASET_FILE = DATA_DIR + 'classification/dataset.csv'


TOY_ANNOTATED_DATA = DATA_DIR + 'toy_dataset.json'
EXTENDED_TOY_DATASET_FILE = DATA_DIR + 'classification/toy_dataset.csv'
AUGMENTED_CORPUS_FILE = {
	"balanced_dataset": "../../gpt3_augmentation/augmented_dataset.json",
}

AAA_DIR = DATA_DIR + 'classification/aaa/'
AAA_FILES = ['f1_o.tsv', 'hashtag_check.tsv', 'corr_a_to_a.tsv', 'corr_n_to_n.tsv', 'quoting_a_to_n.tsv', 'flip_n_to_a.tsv']
HATECHECK_DIR = DATA_DIR + 'classification/hatecheck/'
HATECHECK_FILES = ["counter_quote_nh.tsv",
	"derog_dehum_h.tsv",
	"derog_neg_attrib_h.tsv",
	"ident_pos_nh.tsv",
	"negate_pos_h.tsv",
	"phrase_question_h.tsv",
	"profanity_nh.tsv",
	"ref_subs_sent_h.tsv",
	"slur_h.tsv",
	"spell_char_del_h.tsv",
	"spell_leet_h.tsv",
	"spell_space_del_h.tsv",
	"target_indiv_nh.tsv",
	"threat_dir_h.tsv",
	"counter_ref_nh.tsv",
	"derog_impl_h.tsv",
	"derog_neg_emote_h.tsv",
	"ident_neutral_nh.tsv",
	"negate_neg_nh.tsv",
	"phrase_opinion_h.tsv",
	"profanity_h.tsv",
	"ref_subs_clause_h.tsv",
	"slur_homonym_nh.tsv",
	"slur_reclaimed_nh.tsv",
	"spell_char_swap_h.tsv",
	"spell_space_add_h.tsv",
	"target_group_nh.tsv",
	"target_obj_nh.tsv",
	"threat_norm_h.tsv"
	]

SPLIT_FILES = {
	'train': DATA_DIR + 'train_idx.csv',
	'val': DATA_DIR + 'val_idx.csv',
	'test': DATA_DIR + 'test_idx.csv',
}

EXTENDED_SPLIT_FILES = {
	'train': DATA_DIR + 'classification/train_idx.csv',
	'val': DATA_DIR + 'classification/val_idx.csv',
	'test': DATA_DIR + 'classification/test_idx.csv',
}

LOG_DIR = f'LOG/'

CORPUS_DIR = LOG_DIR + 'corpus/'

EMBEDDINGS_PATH = LOG_DIR + 'embeddings/'

USE_ATTRIBUTES = False
