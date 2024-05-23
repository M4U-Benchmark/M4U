import os
import re
import io
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from utils import (
    setup_seed, 
    reformat_option, 
    PROMPT_TEMPLATE, 
    IMAGE_PALCE_HOLODER,
)
from LLaVA.llava.mm_utils import process_images
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--conv_mode", type=str, default="chatml_direct")
    parser.add_argument("--field", type=str, default='all', help="['all', 'science', 'engineering', 'healthcare']")
    parser.add_argument("--lang", type=str, default="all", help="['all', 'en', 'zh', 'de']")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--result_folder", type=str, default="", help="")
    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.result_folder, exist_ok=True)

    dataset = load_dataset("M4U-Benchmark/M4U")
    eval_fields = ['science', 'engineering', 'healthcare'] if args.field == "all" else args.field.split(",")
    eval_langs = ['en', 'zh', 'de'] if args.lang == "all" else args.lang.split(",")

    model_path = args.model
    conv_mode = args.conv_mode

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
    )


    for lang in eval_langs:
        for field in eval_fields:
            result = {
                "model": model_path,
                "total": 0,
                "record": []
            }
            eval_split = "{}_{}".format(field, lang)
            tgt_path = os.path.join(args.result_folder, eval_split + ".json")
            for item in tqdm(dataset[eval_split]):
                image_files = item["image_files"]
                
                option_str = "\n".join(reformat_option(item["options"]))
                input_str = PROMPT_TEMPLATE[lang].replace("{question}", item["question"]).replace("{options}", option_str)
                image_tags = re.findall(r"<image_\d+>", input_str)
                for idx, img_idx in enumerate(image_tags):
                    input_str = input_str.replace(img_idx, IMAGE_PALCE_HOLODER)

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], input_str)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                images, image_sizes = [], []
                for img_file in image_files:
                    image = Image.open(io.BytesIO(img_file["bytes"])).convert('RGB')
                    images.append(image)
                    image_sizes.append(image.size)
                image_tensor = process_images(images, image_processor, model.config)
                image_tensor = image_tensor[0] if len(image_files) == 1 else image_tensor
                if type(image_tensor) is not list:
                    image_tensor = image_tensor.to(dtype=torch.float16, non_blocking=True).to(model.device)
                else:
                    for idx, img_tensor in enumerate(image_tensor):
                        image_tensor[idx] = img_tensor.to(dtype=torch.float16, non_blocking=True).to(model.device)

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                output_ids = model.generate(
                    input_ids.view(1, -1).to(model.device),
                    images=image_tensor,
                    image_sizes=image_sizes if len(image_sizes) > 1 else image_sizes[0],
                    do_sample=False,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=10,
                    use_cache=True
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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
                json.dump(result, f, indent=4)

