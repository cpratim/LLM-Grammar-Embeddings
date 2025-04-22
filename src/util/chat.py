from warnings import filterwarnings
import logging
from util.data import load_grammar_data

from tqdm import tqdm
from random import shuffle
from transformers import pipeline
import torch
import numpy as np
from pprint import pprint

logging.getLogger("transformers").setLevel(logging.ERROR)
filterwarnings("ignore")

np.random.seed(42)

__system_message_multiple_choice_grammar = """You are a helpful assistant that will help me figure out which sentence is grammatically correct and native-like.
You should only output the number of the correct sentence, '1' if the first sentence is correct, '2' if the second sentence is correct.

ONLY OUTPUT THE NUMBER OF THE CORRECT SENTENCE, NOTHING ELSE.

For example:
Which sentence is more grammatical and native-like?
1) The cat is on the mat.
2) The mat on is the cat.

2"""

__system_message_binary_grammar = """You are a helpful assistant that will help me figure out if a sentence is grammatically correct and native-like. 
You should only output '1' if the sentence is grammatically correct and native-like, and '0' if it is not.

ONLY OUTPUT '1' OR '0', NOTHING ELSE.

For example:
Is this sentence grammatically correct?
1) The cat is on the mat.

1"""

__question_format_multiple_choice_grammar = """Which sentence is more grammatical and native-like? 
1) {sentence_1} 
2) {sentence_2}

ONLY OUTPUT THE NUMBER OF THE CORRECT SENTENCE (1 OR 2), NOTHING ELSE."""

__question_format_binary_grammar = """Is this sentence grammatically correct? 
1) {sentence_1}

ONLY OUTPUT '1' OR '0', NOTHING ELSE."""

__system_message_multiple_choice_arithmetic = """You are a helpful assistant that will help me determine which of two arithmetic expressions is correct.

ONLY OUTPUT THE NUMBER OF THE CORRECT EXPRESSION (1 OR 2), NOTHING ELSE.
For example:
Which of the following arithmetic expressions is correct?
1) 2 + 2 = 4
2) 2 + 2 = 5

2"""

__question_format_multiple_choice_arithmetic = """Which of the following arithmetic expressions is correct?
1) {expression_1}
2) {expression_2}

ONLY OUTPUT THE NUMBER OF THE CORRECT EXPRESSION (1 OR 2), NOTHING ELSE."""

__system_message_binary_arithmetic = """You are a helpful assistant that will help me determine if an arithmetic expression is correct.

ONLY OUTPUT '1' OR '0', NOTHING ELSE.
For example:
Is this arithmetic expression correct?
1) 2 + 2 = 4

1"""

__question_format_binary_arithmetic = """Is this arithmetic expression correct?
1) {expression_1}

ONLY OUTPUT '1' OR '0', NOTHING ELSE."""


def parse_output(output):
    if output not in ["0", "1", "2"]:
        print(f"Failed to parse output: {output}")
        output = "*"
    return output


def load_LLM_messages_multiple_choice(data, n_shot=5, grammar=True, randomize=False):
    if grammar:
        system_message = __system_message_multiple_choice_grammar
        question_format = __question_format_multiple_choice_grammar
    else:
        system_message = __system_message_multiple_choice_arithmetic
        question_format = __question_format_multiple_choice_arithmetic

    messages = [{"role": "system", "content": system_message}]
    if randomize:
        grammar_data = np.random.permutation(grammar_data)
    for idx in range(n_shot):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": question_format.format(
                        sentence_1=data[idx]["good"], sentence_2=data[idx]["bad"]
                    ),
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": question_format.format(
                        sentence_1=data[idx]["bad"], sentence_2=data[idx]["good"]
                    ),
                },
                {"role": "assistant", "content": "2"},
            ]
        )
    return messages


def load_LLM_messages_binary(data, n_shot=5, grammar=True, randomize=False):
    if grammar:
        system_message = __system_message_binary_grammar
        question_format = __question_format_binary_grammar
    else:
        system_message = __system_message_binary_arithmetic
        question_format = __question_format_binary_arithmetic
    messages = [{"role": "system", "content": system_message}]
    if randomize:
        shuffle(data)
    for idx in range(n_shot):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": question_format.format(sentence_1=data[idx]["good"]),
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": question_format.format(sentence_1=data[idx]["bad"]),
                },
                {"role": "assistant", "content": "0"},
            ]
        )
    return messages


def get_mutiple_choice_prompt(tokenizer, good, bad, grammar=True, n_shot=1):
    template = load_LLM_messages_multiple_choice(grammar, n_shot=n_shot)
    if grammar:
        question_format = __question_format_multiple_choice_grammar
    else:
        question_format = __question_format_multiple_choice_arithmetic
    unif = np.random.uniform(0, 1)
    sentences = [good, bad] if unif < 0.5 else [bad, good]
    template.append(
        {
            "role": "user",
            "content": question_format.format(
                sentence_1=sentences[0], sentence_2=sentences[1]
            ),
        }
    )
    formatted_chat = tokenizer.apply_chat_template(
        template, tokenize=False, add_generation_prompt=False
    )
    answer = "1" if unif < 0.5 else "2"
    return formatted_chat, answer


def get_binary_prompt(tokenizer, good, bad, grammar=True, n_shot=1):
    template = load_LLM_messages_binary(grammar, n_shot=n_shot)
    if grammar:
        question_format = __question_format_binary_grammar
    else:
        question_format = __question_format_binary_arithmetic
    unif = np.random.uniform(0, 1)
    sentences = [good, bad] if unif < 0.5 else [bad, good]
    template.append(
        {"role": "user", "content": question_format.format(sentence_1=sentences[0])}
    )
    formatted_chat = tokenizer.apply_chat_template(
        template, tokenize=False, add_generation_prompt=False
    )
    answer = "1" if unif < 0.5 else "0"
    return formatted_chat, answer


def load_llm_generations(
    data, model, tokenizer, device, n_shot, type_="mcq", grammar=True
):

    generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )
    generations = []
    for _, example in enumerate(tqdm(data)):
        if type_ == "mcq":
            formatted_chat, answer = get_mutiple_choice_prompt(
                tokenizer, example["good"], example["bad"], grammar, n_shot
            )
        elif type_ == "binary":
            formatted_chat, answer = get_binary_prompt(
                tokenizer, example["good"], example["bad"], grammar, n_shot
            )
        llm_output = generation_pipeline(
            formatted_chat,
            return_full_text=False,
            max_new_tokens=10,
            do_sample=False,
        )
        generations.append(
            {
                "good_sentence": example["good"],
                "bad_sentence": example["bad"],
                "answer": answer,
                "output": parse_output(llm_output[0]["generated_text"].strip()),
            }
        )
    return generations


def get_eos_states(model, tokenizer, device, formatted_chat):
    tokens = tokenizer(formatted_chat, return_tensors="pt").to(device)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        forward_pass = model(**tokens, output_hidden_states=True, return_dict=True)
    return [s[:, -1, :] for s in forward_pass.hidden_states]


def load_raw_embeddings_dataset(
    data, model, tokenizer, device, types_=["mcq", "bin", "raw"], grammar=True
):

    embeddings = {t: [] for t in types_}
    for _, example in enumerate(tqdm(data)):
        if "mcq" in types_:
            formatted_chat, answer = get_mutiple_choice_prompt(
                tokenizer, example["good"], example["bad"], grammar
            )
            embeddings["mcq"].append(
                {
                    "states": get_eos_states(model, tokenizer, device, formatted_chat),
                    "answer": answer,
                }
            )
        elif "bin" in types_:
            formatted_chat, answer = get_binary_prompt(
                tokenizer, example["good"], example["bad"], grammar
            )
            embeddings["bin"].append(
                {
                    "states": get_eos_states(model, tokenizer, device, formatted_chat),
                    "answer": answer,
                }
            )
        elif "raw" in types_:
            embeddings["raw"].append(
                {
                    "good_states": get_eos_states(
                        model, tokenizer, device, example["good"]
                    ),
                    "bad_states": get_eos_states(
                        model, tokenizer, device, example["bad"]
                    ),
                }
            )
    return embeddings
