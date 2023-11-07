# This code referred to the rating code of AlpaGasus(https://github.com/gpt4life/alpagasus/blob/main/rating/chatgpt_rating.py)

import os
import argparse
import json
import time
import openai
import tiktoken
from tqdm import tqdm
from tqdm import notebook
from huggingface_hub import HfApi
from datasets import load_dataset
import ipywidgets as widgets
from IPython.display import display

# import shortuuid
import asyncio
from typing import Any
import logging

import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            request_timeout=60
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="Cartinoe5930/KoRAE_original",
        help="The path to the dataset. You can load the HF and json both types of datasets."
    )
    parser.add_argument(
        "--i",
        type=int,
        help="The path to the dataset. You can load the HF and json both types of datasets."
    )
    parser.add_argument(
        "--rating_type",
        type=str,
        required=True,
        help="You can choose the rating type between English and Korean. Options: 'en', 'ko'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rating/result/",
        help="the output directory to save the rated data"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="the batch size to call the ChatGPT."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True
    )
    
    return parser.parse_args()

def parse_score(review, rating_type):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:
        if rating_type == "en":
            if('score:' in review):
                score = float(review.split('score:')[1].split('\n')[0])
            elif('Score:' in review):
                score = float(review.split('Score:')[1].strip('\n')[0])
            else:           
                logger.error(
                    f"{e}\nContent: {review}\n" "You must manually fix the score pair."
                )
                score = -1

        elif rating_type == "ko":
            if('점수:' in review):
                score = float(review.split('점수:')[1].split('\n')[0])
            else:
                logger.error(
                    f"{e}\nContent: {review}\n" "You must manually fix the score pair."
                )
                score = -1
    
    return score

def count_tokens_model(data_list):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for data in data_list:
        a = data[0]["content"] + data[1]["content"]
        token_len = len(encoding.encode(a)) + 256
        if token_len > 4000:
            return "gpt-3.5-turbo-16k-0613"
            
    return "gpt-3.5-turbo-0613"

def process_template(data, template_formats):
    # We do not include prompt of dataset to prevent the misleading of rating.
    if data["input"]:
        result = template_formats["no_prompt_input"].format_map(data) + data["output"] + "\n\n"
    else:
        result = template_formats["no_prompt_no_input"].format_map(data) + data["output"] + "\n\n"

    return result

def process_output(predictions, rating_data, template_formats, args, iteration):
    outputs = []
    for idx, prediction in tqdm(enumerate(predictions)):
        review = prediction['choices'][0]['message']['content']
        score = parse_score(review, args.rating_type)
        instruction = process_template(rating_data[iteration+idx], template_formats)
        meta_output = {
                    "instruction": instruction,
                    "review": review,
                    "score": score
                }
        outputs.append(meta_output)
        
    return outputs

if __name__ == "__main__":
    args = args_parse()
    openai.api_key = args.api_key
    message_list = []

    # Load Dataset
    if "json" in args.data_path:
        rating_data = json.load(open(args.data_path))
    else:
        rating_data = load_dataset(args.data_path, split="train")

    # Load system prompt & user prompt for ChatGPT
    if args.rating_type == "ko":
        system_prompt = "다음에 표시되는 명령어와 주어진 입력에 대한 AI 어시스턴트의 성능에 대한 피드백을 요청합니다."
        user_prompt = "지시 및 입력에 대한 응답의 정확성과 유용성에 따라 평가해 주세요. 각 어시스턴트는 0~10점 척도로 점수를 받게 되며, 점수가 높을수록 정확도와 유용성이 높다는 것을 나타냅니다. 먼저 점수를 나타내는 값을 한 줄로 출력해 주세요. 다음 줄에는 편견이 개입되지 않도록 평가에 대한 포괄적인 설명을 입력해 주세요. \n\n"
    elif args.rating_type == "en":
        system_prompt = "We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following."
        user_prompt = "Please rate according to the accuracy and the helpfulness of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 10, where a higher score indicates higher level of the accuracy and helpfulness. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    else:
        raise ValueError("The rating type should be 'en' or 'ko'.")

    with open("templates/KoRAE_template.json", "r", encoding="UTF-8") as f:
        prompt_templates = json.load(f)

    for n in range(len(rating_data)):
        if n < args.i:
            message_list.append("none")
        else:
            eval_prompt = process_template(data=rating_data[n], template_formats=prompt_templates)
            eval_prompt += user_prompt
            message = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": eval_prompt,
                },
            ]
            message_list.append(message)
    
    predictions = []
    i = args.i
    wait_base = 5
    retry = 0
    batch_size = args.batch_size
    pbar = notebook.tqdm(total=len(message_list)-i)
    
    while(i < len(message_list)):
        try:
            if (i + batch_size) > len(message_list):
                mes_list = message_list[i:-1]
            else:
                mes_list = message_list[i:i+batch_size]
            model_name = count_tokens_model(mes_list)
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=mes_list,
                    model=model_name,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    top_p=1.0,
                )
            )
            predictions += batch_predictions
            i += batch_size
            pbar.update(batch_size)
            time.sleep(3)
        except openai.error.Timeout as e:
            retry += 1
            print("OpenAI API Timeout error was occurred in batch size between ", i, i+batch_size)
            print("retry number: ", retry)
            time.sleep(wait_base)
        except TimeoutError:
            retry += 1
            print("TimeoutError was occurred in batch size between ", i, i+batch_size)
            print("retry number: ", retry)
            time.sleep(wait_base)
        except:
            retry += 1
            print("Batch error: ", i, i+int(args.batch_size))
            print("retry number: ", retry)
            time.sleep(wait_base)
            
        if i % 1000 == 0 and i != args.i:
            output_path = args.output_dir + "rated_KoRAE_" + str(i) + ".json"
            if not os.path.isfile(output_path):
                outputs = process_output(predictions=predictions, rating_data=rating_data, template_formats=prompt_templates, args=args, iteration=i-1000)

                with open(args.output_dir + "rated_KoRAE_" + str(i) + ".json", "x", encoding="UTF-8") as output_review_file:
                    json.dump(outputs, output_review_file, indent=4, ensure_ascii=False)

                predictions = []
                args.i += batch_size
               
    pbar.close()

    iteration = (i // 1000) * 1000
    outputs = process_output(predictions=predictions, rating_data=rating_data, template_formats=prompt_templates, args=args, iteration=iteration)

    with open(args.output_dir + "rated_KoRAE_final.json", "x", encoding="UTF-8") as output_review_file:
        json.dump(outputs, output_review_file, indent=4, ensure_ascii=False)