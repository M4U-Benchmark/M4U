import os
import re
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import requests
import base64
import traceback
from utils import (
    setup_seed, 
    reformat_option, 
    PROMPT_TEMPLATE, 
    IMAGE_PALCE_HOLODER,
)

api_key = "/your/api/key"


def generate_requests(model, content):
   headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
   payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1024
   }
   response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
   return response.json()


def generate_reponse(model, content, max_try=100):
   loop_num = 0
   while(True):
      try:
         request = generate_requests(model, content)
      except:
         traceback.print_exc()
         loop_num += 1
         print(loop_num)
         if loop_num > max_try:
            break
         continue
      break
   return request


def reformat_gpt4o_input(input_str, image_files):
    images = [base64.b64encode(img_file["bytes"]).decode('utf-8') for img_file in image_files]

    content_list = []
    split_list = re.split(r'(<image>)', input_str) 
    for split_content in split_list:
        if split_content == '<image>':
            try:
                img = images.pop(0)
            except:
                img = img
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"}
                })
        else:
            content_list.append({
                "type": "text",
                "text": split_content
            })
    return content_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--field", type=str, default="all", help="['all', 'science', 'engineering', 'healthcare']")
    parser.add_argument("--lang", type=str, default="all", help="['all', 'en', 'zh', 'de']")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--result_folder", type=str, default="./gpt4o", help="")
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.result_folder, exist_ok=True)

    eval_fields = ['science', 'engineering', 'healthcare'] if args.field == "all" else args.field.split(",")
    eval_langs = ['en', 'zh', 'de'] if args.lang == "all" else args.lang.split(",")

    model = args.model
    dataset = load_dataset("M4U-Benchmark/M4U")

    for lang in eval_langs:
        for field in eval_fields:
            result = {
                "model": model,
                "total": 0,
                "record": []
            }
            eval_split = "{}_{}".format(field, lang)
            tgt_path = os.path.join(args.result_folder, eval_split + ".json")

            if os.path.exists(tgt_path):
                result = json.load(open(tgt_path, "r"))

            idx_save, begin = 0, result['total']
            for item in tqdm(dataset[eval_split]):
                idx_save += 1
                if idx_save <= begin:
                    continue

                option_str = "\n".join(reformat_option(item["options"]))
                input_str = PROMPT_TEMPLATE[lang].replace("{question}", item["question"]).replace("{options}", option_str)
                image_tags = re.findall(r"<image_\d+>", input_str)
                for idx, img_idx in enumerate(image_tags):
                    input_str = input_str.replace(img_idx, IMAGE_PALCE_HOLODER)  
                
                content_list = reformat_gpt4o_input(input_str, item["image_files"])
                try:
                    response = generate_reponse(model, content_list)
                    outputs = response["choices"][0]['message']['content']
                except:
                    outputs = 'error'

                result["total"] += 1
                result["record"].append({
                    "question": item["question"],
                    "options": item["options"],
                    "image_type": item["image_type"],
                    "cross_lingual": item["cross_lingual"],
                    "gt": item["answer"],
                    "predict": outputs,
                })
                with open(tgt_path, "w") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            

            
            
            



