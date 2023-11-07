from datasets import load_dataset, Dataset

original_KoRAE = load_dataset("Cartinoe5930/KoRAE_original", split="train")
rated_KoRAE = load_dataset("Cartinoe5930/KoRAE_rated", split="train")

original_KoRAE = original_KoRAE.add_column("score", rated_KoRAE["score"])

result = []
for data in original_KoRAE:
    if float(data["score"]) >= 8.5:
        result.append(data)

dataset = Dataset.from_list(result)

dataset.push_to_hub("Cartinoe5930/KoRAE", use_auth_token="")