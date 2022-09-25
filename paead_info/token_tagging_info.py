from .info import *
from .intent_slot_info import *
import os


TT_HIDDEN_SIZE = 300
TT_EMBEDDING_SIZE = 300
TT_NUM_LAYERS = 1
TT_NUM_CLASSES = len(ONTOLOGY) - 5  # Excluding the Intents
TT_MAX_EPOCHS = 30 # 10000
TT_BATCH_SIZE = 1
TT_LEARNING_RATE = 0.001
TT_NEGATIVE_SAMPLES = 0
TT_THRESHOLD = 0.8
TT_DROPOUT = 0.5
PAD_TOKEN = '<pad>'
TT_PRINT_EVERY = 10  #500
TT_OPTIMIZER = 'adam'
TT_OUTPUT_NODES = 'ovo'  # Legal options: powerset, ovo

TT_EXPERIMENT_NAME = f'tokentag_h{TT_HIDDEN_SIZE}_emb{TT_EMBEDDING_SIZE}_lay{TT_NUM_LAYERS}_lr{TT_LEARNING_RATE}_bs{TT_BATCH_SIZE}_ns{TT_NEGATIVE_SAMPLES}_th{TT_THRESHOLD}_do{TT_DROPOUT}_{TT_OPTIMIZER}_{TT_OUTPUT_NODES}'
if not USE_ATTRIBUTES:
	TT_EXPERIMENT_NAME += '_noAttributes'
TT_MODEL_DIR = LOG_DIR + 'tokentag/'
TT_MODEL_FILE = TT_MODEL_DIR + f'{TT_EXPERIMENT_NAME}.pt'

TT_RES_FILE = TT_MODEL_DIR + 'scores.tsv'

TT_PREDICTIONS_FILE = TT_MODEL_DIR + f'{TT_EXPERIMENT_NAME}.res.tsv'

TT_AAA_FILES = TT_MODEL_DIR + f'{TT_EXPERIMENT_NAME}_aaa_answer_files/'
if not os.path.exists(TT_AAA_FILES):
	os.makedirs(TT_AAA_FILES)
