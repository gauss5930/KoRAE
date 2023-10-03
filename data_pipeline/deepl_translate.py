import deepl
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import time

def args_parse():
    parser  = argparse.ArgumentParser()
    parser.add_argument(
        "--API_KEY",
        type=str,
        help="Your DeepL API KEY"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="You can choose the type of dataset. There are two options: 'hf' - HuggingFace Dataset / 'json' - JSON Dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path of dataset"
    )

    return parser.parse_args()

def dataset_load(data_type, data_path):
    dataset = []
    if data_type == "hf":
        loaded_dataset = load_dataset(data_path, split="train")
        i = 0
        for line in loaded_dataset:
            if i == 0:
                columns = list(line.keys())
                i += 1
            dataset.append(line)
    elif data_type == "json":
        with open(data_path, "r") as f:
            dataset = json.load(f)
        columns = list(dataset[0].keys())
    else:
        raise ValueError("The '--data_type' should be 'hf' or 'json'!!")
    
    return dataset, columns

def deepl_api(input_text, auth_key):
    translator = deepl.Translator(auth_key)

    try:
        result = translator.translate_text(input_text, target_lang="KO")
    except:
        print("Retrying due to the error of DeepL API")
        time.sleep(5)
        return deepl_api(input_text, auth_key)

    return result.text

def type_cls(input_data):
    if "```" in input_data:
        return "code"
    elif len(input_data.split("$")) % 2 == 1 and len(input_data.split("$")) >= 5:
        return "math"
    else:
        return None

type_list = {"math": ["$", "ABC"], "code": ["```", "BLOCKED_CODE"]}

def data_process(input_data, data_type):
    splited = input_data.split(type_list[data_type][0])
    blocked_list = []

    for i in range(len(splited)):
        if i % 2 == 1:
            blocked_list.append(type_list[data_type][0] + splited[i] + type_list[data_type][0])
            splited[i] = type_list[data_type][1]

    input_text = " ".join(splited)

    output = deepl_api(input_text)

    output = output.split()

    for i in range(len(output)):
        if output[i] == type_list[data_type][1]:
            output[i] = blocked_list[0]
            blocked_list.pop(0)

    output = " ".join(output)

    return output

if __name__ == "__main__":
    args = args_parse()

    dataset, columns = dataset_load(args.data_type, args.data_path)

    for line in tqdm(range(len(dataset))):
        translated_text = {}
        for column in columns:
            input_data = line[column]

            data_type = type_cls(input_data=input_data)

            translated_text[column] = data_process(input_data=input_data, data_type=data_type)

        with open("data_pipleine/ko-open-wyvern.json", "a") as f:
            json.dump(translated_text, f, indent=4)

        f.close()

    print("Translation is all done!!")