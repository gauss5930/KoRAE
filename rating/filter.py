from datasets import load_dataset, Dataset
import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rated_data",
        type=str,
        default="Cartinoe5930/KoRAE_rated_filtered"
    )
    parser.add_argument(
        "--score_criteria",
        type=float,
        default=8.5,
        help="The score criteria for filtering"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The HuggingFace Hub path to upload the filtered dataset"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Your HuggingFace Access token"
    )

if __name__=="__main__":
    args = args_parse()

    print("Downloading rated dataset.")
    rated_KoRAE = load_dataset(args.rated_data, split="train")

    result = []
    print(f"Filtering rated dataset that higher than {args.score_criteria}")
    for data in rated_KoRAE:
        if float(data["score"]) >= args.score_criteria:
            result.append(data)

    dataset = Dataset.from_list(result)

    dataset.push_to_hub(args.output_dir, use_auth_token=args.hf_token)