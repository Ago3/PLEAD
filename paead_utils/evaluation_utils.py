from collections import Counter
import string


def jaccard_similarity(tokenizedStrings):
	tokenizedStrings = [w.translate(str.maketrans(' ' * len(string.punctuation), string.punctuation)) for ann in tokenizedStrings for w in ann]
	counts = [Counter(ann) for ann in tokenizedStrings]
	intersection = counts[0]
	for i in range(1, len(counts)):
		intersection = intersection & counts[i]
	common_tokens = sum(intersection.values()) * len(tokenizedStrings)
	tot_tokens = sum([len(list(cc.elements())) for cc in counts])
	assert (common_tokens / tot_tokens) >= 0, 'Negative jaccard similarity found'
	return common_tokens / tot_tokens


def dict_symmetric_difference(a, b):
	return {k: a[k] if k in a else b[k] for k in a.keys() ^ b.keys()}
