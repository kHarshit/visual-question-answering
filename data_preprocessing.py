import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import argparse
import yaml
from typing import Text
import logging

def load_config(config_path: Text) -> dict:
    """Load the YAML configuration file."""
    with open(config_path, 'r') as conf_file:
        return yaml.safe_load(conf_file)

def load_qa_data(file_path: Text) -> list:
    """Load question-answer pairs from the specified file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def process_qa_pairs(qa_data: list, question_col: Text, answer_col: Text, image_col: Text) -> pd.DataFrame:
    """Process the raw question-answer pairs into a DataFrame."""
    image_pattern = re.compile(r"( (in |on |of )?(the |this )?(image\d*) \? )")
    
    records = []
    for i in range(0, len(qa_data), 2):
        match = image_pattern.search(qa_data[i])
        if match:
            img_id = match.group(4)
            question = qa_data[i].replace(match.group(0), "").strip()
            answer = qa_data[i+1].strip()
            records.append({question_col: question, answer_col: answer, image_col: img_id})
    
    return pd.DataFrame(records)

def create_answer_space(df: pd.DataFrame, answer_col: Text) -> list:
    """Create a sorted list of all possible unique answers."""
    answers = []
    for ans in df[answer_col]:
        answers.extend(ans.replace(" ", "").split(",") if "," in ans else [ans])
    
    return sorted(set(answers))

def save_answer_space(answer_space: list, output_path: Text) -> None:
    """Save the answer space to a specified file."""
    with open(output_path, 'w') as f:
        f.write("\n".join(answer_space))

def split_and_save_dataset(df: pd.DataFrame, dataset_folder: Text, train_filename: Text, eval_filename: Text) -> None:
    """Split the dataset into training and evaluation sets and save them to CSV files."""
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv(os.path.join(dataset_folder, train_filename), index=False)
    test_df.to_csv(os.path.join(dataset_folder, eval_filename), index=False)

def process_daquar_dataset(config_path: Text) -> None:
    """Main function to process the DAQUAR dataset."""
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO)
    
    dataset_folder = config["data"]["dataset_folder"]
    qa_pairs_file = config["data"]["all_qa_pairs_file"]
    question_col = config["data"]["question_col"]
    answer_col = config["data"]["answer_col"]
    image_col = config["data"]["image_col"]
    answer_space_file = config["data"]["answer_space"]
    train_dataset_file = config["data"]["train_dataset"]
    eval_dataset_file = config["data"]["eval_dataset"]
    
    logging.info("Loading question-answer pairs...")
    qa_data = load_qa_data(os.path.join(dataset_folder, qa_pairs_file))
    
    logging.info("Processing raw QnA pairs...")
    df = process_qa_pairs(qa_data, question_col, answer_col, image_col)
    
    logging.info("Creating space of all possible answers...")
    answer_space = create_answer_space(df, answer_col)
    save_answer_space(answer_space, os.path.join(dataset_folder, answer_space_file))
    
    logging.info("Splitting dataset into training and evaluation sets...")
    split_and_save_dataset(df, dataset_folder, train_dataset_file, eval_dataset_file)
    
    logging.info("Dataset processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the DAQUAR dataset.")
    parser.add_argument('--config', dest='config', required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    process_daquar_dataset(args.config)
