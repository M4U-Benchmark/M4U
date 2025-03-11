# M4U: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models

![Multilingual](https://img.shields.io/badge/Task-Multilingual-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green) 
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![GPT-4V](https://img.shields.io/badge/Model-GPT--4V-green)
![Gemini](https://img.shields.io/badge/Model-Gemini-green)

Code for the Paper M4U: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models.

[[Webpage](https://m4u-benchmark.github.io/m4u.github.io/)] [[Paper](http://arxiv.org/abs/2405.15638)] [[Huggingface Dataset](https://huggingface.co/datasets/M4U-Benchmark/M4U)] [[Leaderboard](https://m4u-benchmark.github.io/m4u.github.io/)]

<p align="center">
    <img src="images/logo.png" width=20%"> <br>
  <b>M4U</b>: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models
</p>

## Outlines

- [🎨 M4U-mini](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#-m4u-mini)
  - [Evaluation on M4U-mini](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#evaluation-on-m4u-mini)
  - [Evaluation results](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#evaluation-results)
- [📖 Dataset Usage](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#-dataset-usage)
- [✅ Cite](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#-cite)
- [🧠 Acknowledgments](https://github.com/M4U-Benchmark/M4U/tree/m4u-mini?tab=readme-ov-file#-acknowledgments)

## 🎨 M4U-mini

To support more languages, we have constructed a tiny version of M4U with three additional languages (Japanese, Arabic, and Thai). We randomly selected 5% of the test data and follow our processing pipeline to construct these parts. We plan to expand M4U to include these medium or low-resource languages and more disciplines regrading cultural contexts later.

### Evaluation on M4U-mini

The evaluation pipeline of M4U-mini is consistent with M4U. You can generate the response of GPT-4o on M4U-mini, then calculate the scores following [these instructions](https://github.com/M4U-Benchmark/M4U/tree/main?tab=readme-ov-file#evaluating-openai-models).
```sh
python evaluate_gpt4o.py \
  --model gpt-4o \
  --field all \
  --lang zh,en,de,ar,th,ja \
  --result_folder ./result/M4U-mini/gpt4o
```

### Evaluation results

We evaluate state-of-the-art close-sourced models (GPT-4o, Gemini-1.5-Flash) and open-sourced models (DeepSeek-VL-Chat, LLaVA-NeXT-34B, etc.) on M4U-mini. Detailed results are shown below:

| **#** | **Model**                       | **Method** | **Source**                                                   | **English** | **Chinese** | **German**  | **Japanese** | **Arabic** | **Thai** | **Average** |
| ----- | ------------------------------- | ---------- | ------------------------------------------------------------ | ----------- | ----------- | ---------  | ----------- | ----------- | ---------   | ----------- |
| 1     | GPT-4o     | LMM      | gpt-4o     | 44.9    | 53.7     | 42.4    | 49.1  | 45.2     | 48.8    | 47.3  |
| 2     | InternVL2.5-26B     | LMM      | [LINK](https://huggingface.co/OpenGVLab/InternVL2_5-26B)|  44.2   | 51.3    | 48.1     | 46.4    | 37.6  | 47.3     | 44.2  |
| 3     | Qwen2-VL-7B-Instruct     | LMM      | [LINK](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)|  43.5   | 46.6    | 44.1     | 47.6    | 41.5  | 41.4     | 44.1  |
| 4     | Gemini-1.5-Flash     | LMM      | gemini-1.5-flash |  35.4   | 46.3    | 42.8     | 39.0    | 38.4  | 40.1     | 40.3  |
| 5     | InternVL2.5-8B     | LMM      | [LINK](https://huggingface.co/OpenGVLab/InternVL2_5-8B)|  41.7   | 38.5    | 38.3     | 36.1    | 31.4  | 31.7     | 36.3  |
| 6     | LLaVA-NeXT-34B     | LMM      | [LINK](https://huggingface.co/liuhaotian/llava-v1.6-34b)|  44.1  | 44.2    | 39.0     | 36.0    | 11.4  | 34.0     | 34.8  |
| 7     | Phi-3.5-Vision-Instruct     | LMM      | [LINK](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)|  34.3  | 27.2    | 33.4     | 30.4    | 31.7  | 30.9     | 31.3  |
| 8     | DeepSeek-VL-Chat     | LMM      | [LINK](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)|  35.4  | 33.6    | 35.0     | 32.1    | 24.8  | 25.4     | 31.0  |


## 📖 Dataset Usage

The format of M4U-mini is consistent with M4U. First, make sure that you have successfully setup:

```sh
pip install datasets
```

Then you can easily download this dataset from [Huggingface](https://huggingface.co/datasets/M4U-Benchmark/M4U-mini).
```python
from datasets import load_dataset

dataset = load_dataset("M4U-Benchmark/M4U-mini")
```

## ✅ Cite

If you find **M4U** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{wang2024m4u,
  title={M4U: Evaluating Multilingual Understanding and Reasoning for Large Multimodal Models},
  author={Hongyu Wang and Jiayu Xu and Senwei Xie and Ruiping Wang and Jialin Li and Zhaojie Xie and Bin Zhang and Chuyan Xiong and Xilin Chen},
  month={May},
  year={2024}
}
```

## 🧠 Acknowledgments

Some implementations in M4U are either adapted from or inspired by the [MMMU](https://github.com/MMMU-Benchmark/MMMU) repository and the [MathVista](https://github.com/lupantech/MathVista) repository.
