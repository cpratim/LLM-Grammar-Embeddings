from warnings import filterwarnings
import logging
from util.data import load_grammar_data
from random import sample

from tqdm import tqdm
from random import shuffle
from transformers import pipeline
import torch
import numpy as np
from pprint import pprint
from util.model import clear_cuda_cache

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

__question_format_multiple_choice_grammar = """Which sentence is more grammatically correct? 
1) {sentence_1} 
2) {sentence_2}"""

__question_format_binary_grammar = """Is this sentence grammatically correct? 
1) {sentence_1}"""

__system_message_multiple_choice_arithmetic = """You are a helpful assistant that will help me determine which of two arithmetic expressions is correct.

ONLY OUTPUT THE NUMBER OF THE CORRECT EXPRESSION (1 OR 2), NOTHING ELSE.
For example:
Which of the following arithmetic expressions is correct?
1) 2 + 2 = 4
2) 2 + 2 = 5

1"""

__question_format_multiple_choice_arithmetic = """Which of the following arithmetic expressions is correct?
1) {sentence_1}
2) {sentence_2}"""

__system_message_binary_arithmetic = """You are a helpful assistant that will help me determine if an arithmetic expression is correct.

ONLY OUTPUT '1' OR '0', NOTHING ELSE.
For example:
Is this arithmetic expression correct?
1) 2 + 2 = 4

1"""

__question_format_binary_arithmetic = """Is this arithmetic expression correct?
1) {sentence_1}"""

__system_message_multiple_choice_word_problems = """You are a helpful assistant that will help me figure out which statement is correct.
You should only output the number of the correct statement, '1' if the first statement is correct, '2' if the second statement is correct.

ONLY OUTPUT THE NUMBER OF THE CORRECT SENTENCE, NOTHING ELSE.

For example:
Which statement is correct?
1) I ate 2 apples and 2 oranges, the total number of fruits I ate is 4.
2) I ate 2 apples and 2 oranges, the total number of fruits I ate is 5.

1"""

__system_message_binary_word_problems = """You are a helpful assistant that will help me figure out if a statement is correct.
You should only output '1' if the statement is correct, and '0' if it is not.

ONLY OUTPUT '1' OR '0', NOTHING ELSE.

For example:
Is this statement correct?
1) I ate 2 apples and 2 oranges, the total number of fruits I ate is 4.

1"""

__question_format_binary_word_problems = """Is this statement numerically correct?
1) {sentence_1}"""

__question_format_multiple_choice_word_problems = """Which statement is numerically correct?
1) {sentence_1}
2) {sentence_2}"""   


def parse_output(output):
    if output not in ["0", "1", "2"]:
        print(f"Failed to parse output: {output}")
        output = "*"
    return output


def load_LLM_messages_multiple_choice(
    shots, type_=True, randomize=False, use_system_message=True
):
    if type_ == "grammar":
        system_message = __system_message_multiple_choice_grammar
        question_format = __question_format_multiple_choice_grammar
    elif type_ == "arithmetic":
        system_message = __system_message_multiple_choice_arithmetic
        question_format = __question_format_multiple_choice_arithmetic
    elif type_ == "word_problems":
        system_message = __system_message_multiple_choice_word_problems
        question_format = __question_format_multiple_choice_word_problems

    messages = []
    if use_system_message:
        messages.append({"role": "system", "content": system_message})
    if randomize:
        shots = np.random.permutation(shots)
    for idx in range(0, len(shots)):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": question_format.format(
                        sentence_1=shots[idx]["good"], sentence_2=shots[idx]["bad"]
                    ),
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": question_format.format(
                        sentence_1=shots[idx]["bad"], sentence_2=shots[idx]["good"]
                    ),
                },
                {"role": "assistant", "content": "2"},
            ]
        )
    return messages


def load_LLM_messages_binary(
    shots, type_=True, randomize=False, use_system_message=True
):
    if type_ == "grammar":
        system_message = __system_message_binary_grammar
        question_format = __question_format_binary_grammar
    elif type_ == "arithmetic":
        system_message = __system_message_binary_arithmetic
        question_format = __question_format_binary_arithmetic
    elif type_ == "word_problems":
        system_message = __system_message_binary_word_problems
        question_format = __question_format_binary_word_problems
    messages = []
    if use_system_message:
        messages.append({"role": "system", "content": system_message})
    if randomize:
        shuffle(shots)
    for idx in range(0, len(shots)):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": question_format.format(sentence_1=shots[idx]["good"]),
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": question_format.format(sentence_1=shots[idx]["bad"]),
                },
                {"role": "assistant", "content": "0"},
            ]
        )
    return messages


def get_mutiple_choice_prompt(
    sentence_1, sentence_2, type_="grammar"
):
    if type_ == "grammar":
        question_format = __question_format_multiple_choice_grammar
    elif type_ == "arithmetic":
        question_format = __question_format_multiple_choice_arithmetic
    elif type_ == "word_problems":
        question_format = __question_format_multiple_choice_word_problems

    chat = question_format.format(sentence_1=sentence_1, sentence_2=sentence_2)
    return chat

def get_binary_prompt(
    sentence_1, type_="grammar"
):
    if type_ == "grammar":
        question_format = __question_format_binary_grammar
    elif type_ == "arithmetic":
        question_format = __question_format_binary_arithmetic
    elif type_ == "word_problems":
        question_format = __question_format_binary_word_problems
    chat = question_format.format(sentence_1=sentence_1)
    return chat


def load_llm_generations(
    data,
    model,
    tokenizer,
    device,
    n_shot,
    type_="mcq",
    problem_type="grammar",
    use_system_message=True,
):
    shots = sample(data, n_shot)
    generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map=device
    )
    generations = []
    for _, example in enumerate(tqdm(data)):
        if type_ == "mcq":
            formatted_chat, answer = get_mutiple_choice_prompt(
                shots,
                tokenizer,
                example["good"],
                example["bad"],
                problem_type,
                use_system_message,
            )
        elif type_ == "bin":
            formatted_chat, answer = get_binary_prompt(
                shots,
                tokenizer,
                example["good"],
                example["bad"],
                problem_type,
                use_system_message,
            )
        llm_output = generation_pipeline(
            formatted_chat,
            return_full_text=False,
            max_new_tokens=10,
            do_sample=False,
        )
        # print(example)
        # print(formatted_chat)
        # print()
        generations.append(
            {
                "good": example["good"],
                "bad": example["bad"],
                "answer": answer,
                "output": parse_output(llm_output[0]["generated_text"].strip()),
            }
        )
    return generations


def get_eos_states_batch(model, tokenizer, device, formatted_chat):
    tokens = tokenizer(formatted_chat, return_tensors="pt", padding=True).to(device)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        forward_pass = model(**tokens, output_hidden_states=True, return_dict=True)
    ans = np.array(
        [s[:, -1, :].detach().cpu().numpy() for s in forward_pass.hidden_states]
    )
    return ans.transpose(1, 0, 2)


def get_eos_states(model, tokenizer, device, formatted_chat, get_attentions=False):
    tokens = tokenizer(formatted_chat, return_tensors="pt").to(device)
    with torch.no_grad():
        forward_pass = model(**tokens, output_hidden_states=True, output_attentions=get_attentions, return_dict=True)
    ans = [s[0, -1, :].detach().cpu().numpy() for s in forward_pass.hidden_states]
    return ans


def get_head_attentions(model, tokenizer, device, formatted_chat):
    tokens = tokenizer(formatted_chat, return_tensors="pt").to(device)
    with torch.no_grad():
        forward_pass = model(**tokens, output_attentions=True, return_dict=True)
    attentions = np.array([s.detach().cpu().numpy() for s in forward_pass.attentions])
    return attentions

def load_attentions_dataset(
    data,
    model,
    tokenizer,
    device,
    n_shot=1,
    types_=["mcq", "bin", "raw"],
    problem_type="grammar",
    use_tqdm=True,
    use_system_message=True,
):
    shots = sample(data, n_shot)
    embeddings = {t: [] for t in types_}
    if use_tqdm:
        iterator = tqdm(enumerate(data))
    else:
        iterator = enumerate(data)
    for _, example in iterator:
        with torch.no_grad():
            if "mcq" in types_:
                formatted_chat, answer = get_mutiple_choice_prompt(
                    example["good"],
                    example["bad"],
                    problem_type,
                    use_system_message=use_system_message,
                )
                embeddings["mcq"].append(
                    {
                        "attentions": get_head_attentions(
                            model, tokenizer, device, formatted_chat
                        ),
                        "answer": answer,
                    }
                )
            if "bin" in types_:
                formatted_chat, answer = get_binary_prompt(
                    shots,
                    tokenizer,
                    example["good"],
                    example["bad"],
                    problem_type,
                    use_system_message=use_system_message,
                )
                embeddings["bin"].append(
                    {
                        "attentions": get_head_attentions(
                            model, tokenizer, device, formatted_chat
                        ),
                        "answer": answer,
                    }
                )
            if "raw" in types_:
                embeddings["raw"].append(
                    {
                        "attentions": get_head_attentions(
                            model, tokenizer, device, example["good"]
                        ),
                        "answer": 1,
                    }
                )
                embeddings["raw"].append(
                    {
                        "attentions": get_head_attentions(
                            model, tokenizer, device, example["bad"]
                        ),
                        "answer": 0,
                    }
                )
        clear_cuda_cache()
    return embeddings


def load_embeddings_dataset(
    data,
    model,
    tokenizer,
    device,
    n_shot=1,
    types_=["mcq", "bin", "raw"],
    problem_type="grammar",
    use_tqdm=True,
    use_system_message=True,
):
    shots = sample(data, n_shot)
    embeddings = {t: [] for t in types_}
    if use_tqdm:
        iterator = tqdm(enumerate(data))
    else:
        iterator = enumerate(data)
    for _, example in iterator:
        with torch.no_grad():
            if "mcq" in types_:
                chat = get_mutiple_choice_prompt(
                    example["good"],
                    example["bad"],
                    problem_type,
                )
                embeddings["mcq"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat
                        ),
                        "answer": 0,
                    }
                )
                chat = get_mutiple_choice_prompt(
                    example["bad"],
                    example["good"],
                    problem_type,
                )
                embeddings["mcq"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat
                        ),
                        "answer": 1,
                    }
                )
            if "bin" in types_:
                chat = get_binary_prompt(
                    example["good"],
                    problem_type,
                )
                embeddings["bin"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat
                        ),
                        "answer": 1,
                    }
                )
                chat = get_binary_prompt(
                    example["bad"],
                    problem_type,
                )
                embeddings["bin"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat
                        ),
                        "answer": 0,
                    }
                )
            if "raw" in types_:
                if problem_type == 'grammar':
                    chat = "Pay attention to the grammatical correctness of the following sentence: {}"
                elif problem_type == 'arithmetic':
                    chat = "Pay attention to the arithmetic correctness of the following expression: {}"
                elif problem_type == 'word_problems':
                    chat = "Pay attention to the numerical correctness of the following statement: {}"
                embeddings["raw"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat.format(example["good"])
                        ),
                        "answer": 1,
                    }
                )
                embeddings["raw"].append(
                    {
                        "states": get_eos_states(
                            model, tokenizer, device, chat.format(example["bad"])
                        ),
                        "answer": 0,
                    }
                )
        clear_cuda_cache()
    return embeddings


def load_embeddings_dataset_batch(
    data,
    model,
    tokenizer,
    device,
    n_shot=1,
    types_=["mcq", "bin", "raw"],
    problem_type="grammar",
    batch_size=25,
    use_tqdm=True,
    use_system_message=True,
):
    shots = sample(data, n_shot)
    embeddings = {t: [] for t in types_}
    if use_tqdm:
        iterator = tqdm(range(0, len(data), batch_size))
    else:
        iterator = range(0, len(data), batch_size)
    for i in iterator:
        batch = data[i : i + batch_size]
        mcq_formatted_chats, mcq_answers = [], []
        if "mcq" in types_:
            for example in batch:
                formatted_chat, answer = get_mutiple_choice_prompt(
                    shots,
                    tokenizer,
                    example["good"],
                    example["bad"],
                    problem_type,
                    use_system_message,
                )
            mcq_formatted_chats.append(formatted_chat)
            mcq_answers.append(answer)
            mcq_eos_states = get_eos_states_batch(
                model, tokenizer, device, mcq_formatted_chats
            )
            embeddings["mcq"].extend(
                [
                    {"states": mcq_eos_states[i], "answer": mcq_answers[i]}
                    for i in range(len(mcq_eos_states))
                ]
            )
        if "bin" in types_:
            bin_formatted_chats, bin_answers = [], []
            for example in batch:
                formatted_chat, answer = get_binary_prompt(
                    shots,
                    tokenizer,
                    example["good"],
                    example["bad"],
                    problem_type,
                    use_system_message,
                )
                bin_formatted_chats.append(formatted_chat)
                bin_answers.append(answer)
                bin_eos_states = get_eos_states_batch(
                    model, tokenizer, device, bin_formatted_chats
                )
                embeddings["bin"].extend(
                    [
                        {"states": bin_eos_states[i], "answer": bin_answers[i]}
                        for i in range(len(bin_eos_states))
                    ]
                )
        if "raw" in types_:
            good_raw_formatted_chats, bad_raw_formatted_chats = [], []
            for example in batch:
                formatted_chat = example["good"]
                good_raw_formatted_chats.append(formatted_chat)
                formatted_chat = example["bad"]
                bad_raw_formatted_chats.append(formatted_chat)
            good_raw_eos_states = get_eos_states_batch(
                model, tokenizer, device, good_raw_formatted_chats
            )
            bad_raw_eos_states = get_eos_states_batch(
                model, tokenizer, device, bad_raw_formatted_chats
            )
            embeddings["raw"].extend(
                [
                    {"states": good_raw_eos_states[i], "answer": 1}
                    for i in range(len(good_raw_eos_states))
                ]
            )
            embeddings["raw"].extend(
                [
                    {"states": bad_raw_eos_states[i], "answer": 0}
                    for i in range(len(bad_raw_eos_states))
                ]
            )
        clear_cuda_cache()

    return embeddings
