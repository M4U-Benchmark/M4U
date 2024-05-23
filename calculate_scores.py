import os
import json
import random
import argparse
from utils import (
    setup_seed,
    process_records,
    print_table
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, default='all', help="['all', 'science', 'engineering', 'healthcare']")
    parser.add_argument("--lang", type=str, default="all", help="['all', 'en', 'zh', 'de']")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--result_folder", type=str, default="", help="")
    args = parser.parse_args()
    
    setup_seed(args.seed)

    eval_fields = ['science', 'engineering', 'healthcare'] if args.field == "all" else args.field.split(",")
    eval_langs = ['en', 'zh', 'de'] if args.lang == "all" else args.lang.split(",")
    
    result = []
    for lang in eval_langs:
        for field in eval_fields:
            context = json.load(open(os.path.join(args.result_folder, "{}_{}.json".format(field, lang)), 'r'))
            ncorrect, total = process_records(context["record"])
            result.append({
                "field": field,
                "language": lang,
                "total": total,
                "ncorrect": ncorrect,
                "acc": ncorrect / total,
            })
    print("*"*10, context["model"], "*"*10)    
    print_table(result)
    
    tgt_path = os.path.join(args.result_folder, "eval_result.json")
    with open(tgt_path, "w") as f:
        json.dump(result, f, indent=4)