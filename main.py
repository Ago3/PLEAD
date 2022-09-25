from paead_pipelines import run_pipeline
import argparse
import sys
from paead_info import LOG_DIR


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--name', type=str, default='exp')
	args = parser.parse_args()
	with open(f'{LOG_DIR}output_log_{args.name}_{args.seed}.txt', 'w+') as out_log:
		sys.stdout = out_log
		#run_pipeline('classification', 'roberta', toy=False, extended_dataset=False, seed=args.seed)
		#run_pipeline('intent_and_slot_filling', 'token_tagging_embeddings', toy=False, seed=args.seed)
		#run_pipeline('intent_and_slot_filling', 'bert_token_tagging', toy=False, seed=args.seed)
		#run_pipeline('intent_and_slot_filling', 'litbart', toy=False, seed=args.seed)
		#run_pipeline('intent_and_slot_filling', 'litmsbart', toy=False, seed=args.seed)
		run_pipeline('intent_and_slot_filling', 'litmsbart_with_slot2intent', toy=False, seed=args.seed)
		# run_pipeline('aaa', 'roberta', toy=False, extended_dataset=False, seed=args.seed)
		# run_pipeline('aaa', 'token_tagging_embeddings', toy=False, extended_dataset=False, seed=args.seed)
		# run_pipeline('aaa', 'litbart', toy=False, extended_dataset=False, seed=args.seed)
		# run_pipeline('aaa', 'litmsbart', toy=False, extended_dataset=False, seed=args.seed)
		# run_pipeline('aaa', 'litmsbart_with_slot2intent', toy=False, extended_dataset=False, seed=args.seed)
		# run_pipeline('slots_to_intent', 'slots_to_intent', toy=False, extended_dataset=False)
		

if __name__ == '__main__':
	main()
