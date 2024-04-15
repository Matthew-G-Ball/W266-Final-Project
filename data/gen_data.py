def main():
    import argparse
    import datasets
    from tqdm import tqdm
    from datasets import load_dataset 
    import pandas as pd
    import random
    import json

    parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split", default=0.2, help="split between train and validation")
    parser.add_argument("--seed", default=2024, help="seed for randomizing split")
    parser.add_argument("--testSize", default=5000, help="set test size")
    args = parser.parse_args()

    PROMPT_DICT = """
    ### Question
    Write an SQL query that answers this question:
    {question}

    ### Context
    The query will run on a database with the following schema:
    {context}
    """
    
    print("Loading Data From HuggingFace: Clinton/Text-to-sql-v1")
    Clinton_dataset = load_dataset("Clinton/Text-to-sql-v1")
    print("Loading Data From HuggingFace: b-mc2/sql-create-context")
    b_mc2_dataset = load_dataset("b-mc2/sql-create-context")
    print("Loading Data From HuggingFace: gretelai/synthetic_text_to_sql")
    gretelai_dataset = load_dataset("gretelai/synthetic_text_to_sql")

    pd_clinton = Clinton_dataset["train"].to_pandas()[['instruction', 'input', 'response']]
    pd_mc2 = b_mc2_dataset["train"].to_pandas()
    pd_gretel = gretelai_dataset["train"].to_pandas()[['sql_prompt','sql_context','sql']]
    pd_clinton.columns = ["question","context","answer"]
    pd_gretel.columns = ["question","context","answer"]

    combined_data = pd.concat([pd_clinton,pd_mc2,pd_gretel],ignore_index=True)

    output_list = []
    for i in tqdm(range(len(combined_data)), desc="Mapping Data Points to Prompt"):
        question = combined_data.iloc[i]["question"]
        context = combined_data.iloc[i]["context"]
        nl = PROMPT_DICT.format_map({"question":question,"context":context})
        code = combined_data.iloc[i]["answer"]
        output_list.append({"nl":nl,"code":code})

    print("Randomizing List Values")
    random.Random(args.seed).shuffle(output_list)
    split_num = int(len(output_list)*args.split)

    with open("train.json","w") as f:
        for x in tqdm(output_list[split_num:], desc="Writing Data to train.json"):
            json.dump(x,f,indent=None,default=str)
            f.write("\n")
        f.close()
    with open("validation.json","w") as f:
        for x in tqdm(output_list[:int(split_num-args.testSize)], desc="Writing Data to validation.json"):
            json.dump(x,f,indent=None,default=str)
            f.write("\n")
        f.close()
    with open("testing.json","w") as f:
        for x in tqdm(output_list[int(split_num-args.testSize):split_num], desc="Writing Data to testing.json"):
            json.dump(x,f,indent=None,default=str)
            f.write("\n")
        f.close()

if __name__ == "__main__":
    main()