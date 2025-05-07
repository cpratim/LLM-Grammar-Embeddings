from random import randint, uniform, shuffle
import json


def create_dataset(n_samples, digit_range=(0, 2)):
    data = []
    for _ in range(n_samples):
        d1, d2 = randint(digit_range[0], digit_range[1]), randint(digit_range[0], digit_range[1])
        a, b = randint(10 ** d1, 10 ** (d1 + 1)), randint(10 ** d2, 10 ** (d2 + 1))
        
        if uniform(0, 1) < 0.5:
            ans = a + b
            data.append({
                'good': f'{a} + {b} = {ans}',
                'bad_p_1': f'{a} + {b} = {ans + 1}',
                'bad_m_1': f'{a} + {b} = {ans - 1}',
                'bag_p_10': f'{a} + {b} = {int(ans * 1.1)}',
                'bag_m_10': f'{a} + {b} = {int(ans * 0.9)}',
                'bad_p_25': f'{a} + {b} = {int(ans * 1.25)}',
                'bad_m_25': f'{a} + {b} = {int(ans * 0.75)}',
                'bad_p_50': f'{a} + {b} = {int(ans * 1.5)}',
                'bad_m_50': f'{a} + {b} = {int(ans * 0.5)}',
                'bad_p_95': f'{a} + {b} = {int(ans * 1.95)}',
                'bad_m_95': f'{a} + {b} = {int(ans * 0.05)}',
            })
        else:
            a, b = max(a, b), min(a, b)
            ans = a - b
            data.append({
                'good': f'{a} - {b} = {ans}',
                'bad_p_1': f'{a} - {b} = {ans + 1}',
                'bad_m_1': f'{a} - {b} = {ans - 1}',
                'bag_p_10': f'{a} - {b} = {int(ans * 1.1)}',
                'bag_m_10': f'{a} - {b} = {int(ans * 0.9)}',
                'bad_p_25': f'{a} - {b} = {int(ans * 1.25)}',
                'bad_m_25': f'{a} - {b} = {int(ans * 0.75)}',
                'bad_p_50': f'{a} - {b} = {int(ans * 1.5)}',
                'bad_m_50': f'{a} - {b} = {int(ans * 0.5)}',
                'bad_p_95': f'{a} - {b} = {int(ans * 1.95)}',
                'bad_m_95': f'{a} - {b} = {int(ans * 0.05)}',
            })

    return data


def fill_in_example(example, a, b, ans):
    num = [a, b, ans]
    ex = ''
    for i, p in enumerate(example.split('-')):
        if i < len(num):
            ex += p + str(num[i])
        else:
            ex += p
    return ex


def load_word_problems_data(data_path: str = "../../data/word_problems/fill_in_simple.json", numerical_data_path: str = "../../data/arithmetic/addition_subtraction.jsonl", margin: int = 50):
    with open(data_path, "r") as f:
        data = json.load(f)
    with open(numerical_data_path, "r") as f:
        numerical_data = [json.loads(line) for line in f]

    shuffle(numerical_data)
    addition_problems, subtraction_problems = [], []
    for example in numerical_data:
        good = example['good']
        bad = example[f'bad_p_{margin}'] if uniform(0, 1) < 0.5 else example[f'bad_m_{margin}']
        ans_good = int(good.split('=')[1])
        ans_bad = int(bad.split('=')[1])

        if '+' in good:
            a, b = int(good.split('+')[0]), int(good.split('+')[1].split('=')[0])
            addition_problems.append((a, b, ans_good, ans_bad))
        else:
            a, b = int(good.split('-')[0]), int(good.split('-')[1].split('=')[0])
            subtraction_problems.append((a, b, ans_good, ans_bad))

    dataset = []
    for i, example in enumerate(data['addition']):
        a, b, ans_good, ans_bad = addition_problems[i]
        dataset.append({
            'good': fill_in_example(example, a, b, ans_good),
            'bad': fill_in_example(example, a, b, ans_bad),
        })

    for i, example in enumerate(data['subtraction']):
        a, b, ans_good, ans_bad = subtraction_problems[i]
        dataset.append({
            'good': fill_in_example(example, a, b, ans_good),
            'bad': fill_in_example(example, a, b, ans_bad),
        })

    shuffle(dataset)

    with open('../../data/word_problems/fill_in_addition_subtraction_simple.jsonl', 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

    return dataset

if __name__ == "__main__":
    # data = create_dataset(1000)
    # # create a jsonl file with the data
    # with open('../../data/arithmetic/addition_subtraction.jsonl', 'w') as f:
    #     for item in data:
    #         f.write(json.dumps(item) + '\n')
    data = load_word_problems_data()