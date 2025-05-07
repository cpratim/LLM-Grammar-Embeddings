from util.data import load_combined_grammar_data

grammar_data = load_combined_grammar_data(n_sample=1000)

fields = set()
linguistics_terms = set()
for example in grammar_data:
    fields.add(example["field"])
    linguistics_terms.add(example["linguistics_term"])

print(fields)
print(linguistics_terms)