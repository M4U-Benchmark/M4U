import torch
import random
import numpy as np
import pandas as pd

IMAGE_PALCE_HOLODER = "<image>"

PROMPT_TEMPLATE = {
    "en": "{question}\n{options}\nAnswer with the option's letter from the given choices directly.",
    "de": "{question}\n{options}\nAntworten Sie direkt mit dem Buchstaben der gegebenen Optionen.",
    "zh": "{question}\n{options}\n直接用给定选项的字母回答."
}

TOTAL_IMAGE_TYPES = [
    'Photographs', 
    'diagrams', 
    'MRI, CT scans and X-rays', 
    'Tables', 
    'Chemical Structures', 
    'maps', 
    'Electrocardiogram', 
    'Plots and Charts', 
    'Technical Blueprints', 
    'Microscopic Images', 
    'graph', 
    'Pathological Images', 
    'Medical Images'
]

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reformat_option(lst):
    str_lst = [str(item) for item in lst]
    str_lst = [f"{chr(65+i)}. {item}" for i, item in enumerate(str_lst)]
    return str_lst


def get_choices_index2ans(options):
    all_choices = [chr(65 + i) for i in range(len(options))]
    index2ans = {}
    for idx, choice in enumerate(all_choices):
        index2ans[choice] = options[idx]
    return all_choices, index2ans


# modified from https://github.com/MMMU-Benchmark/MMMU/blob/fd294deeb28352479ca2da13783bbf1e2fc952cf/eval/utils/eval_utils.py#L10
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A. B. C. D.
            if f'{choice}.' in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = 'Z'
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def process_records(records):
    ncorrect, total = 0, 0
    for idx, rec in enumerate(records):
        all_choices, index2ans = get_choices_index2ans(rec["options"])
        predict = parse_multi_choice_response(
            rec["predict"], 
            all_choices=all_choices,
            index2ans=index2ans,
        )
        total += 1
        ncorrect += (predict.strip().lower() == rec["gt"].strip().lower())
    return ncorrect, total


def print_table(result):
    df = pd.DataFrame(result)
    df['acc'] = (df['acc'] * 100).round(1)
    pivot_table = df.pivot(index='language', columns='field', values='acc')
    pivot_table['average'] = pivot_table.mean(axis=1).round(1)
    average_row = pivot_table.mean(axis=0).round(1)
    average_row.name = 'average'
    pivot_table = pd.concat([pivot_table, pd.DataFrame(average_row).T])
    print(pivot_table)

