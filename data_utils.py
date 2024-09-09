import random
import string
from datasets import DatasetDict, Dataset, load_dataset
import json

# def generate_random_text_data(num_samples, max_length):
#     """Generate random text and labels for a dataset."""
#     data = {
#         "text": ["".join(random.choices(string.ascii_letters + " ", k=random.randint(10, max_length))) for _ in range(num_samples)],
#         "label": [random.randint(0, 1) for _ in range(num_samples)]
#     }
#     return data

def generate_random_text_data(num_samples, max_length):
    """Generate random text and labels for a dataset."""
    data = [
        {
            "text": "".join(random.choices(string.ascii_letters + " ", k=random.randint(10, max_length))),
            "label": random.randint(0, 1)
        }
        for _ in range(num_samples)
    ]
    return data

def save_dataset_to_json(data, file_path):
    """Save generated data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def change_data(ds, data_type):
    """Mock change_data function."""
    # This function can simulate any changes you want to make to the dataset
    return ds

if __name__ == "__main__":
    # Generate random dataset for training and testing
    train_data = generate_random_text_data(num_samples=100, max_length=256)
    test_data = generate_random_text_data(num_samples=20, max_length=256)

    # Save the random datasets to JSON files
    train_file_path = "./train.json"
    test_file_path = "./test.json"

    save_dataset_to_json(train_data, train_file_path)
    save_dataset_to_json(test_data, test_file_path)

    # Now load the dataset using the load_dataset function
    def load_local_dataset(train_file_path, test_file_path):
        """Load the local dataset from JSON files using load_dataset."""
        ds = load_dataset("json", data_files={"train": train_file_path, "test": test_file_path})
        return ds
    
    # Load the saved random dataset
    ds = load_local_dataset(train_file_path, test_file_path)

    # Print a sample to verify the data is loaded correctly
    print(f"Lenght of training dataset: {len(ds['train'])}")
    print(f"Sample from training dataset: {ds['train'][0]}")

    # # # Save to local
    # # with open("train.json", "w") as f:
    # #     json.dump(train_data, f)

    # # with open("test.json", "w") as f:
    # #     json.dump(test_data, f)

    # # Convert the random data into Hugging Face Dataset format
    # from datasets import DatasetDict, Dataset
    # from transformers import AutoTokenizer

    # ds = DatasetDict({
    #     "train": Dataset.from_dict(train_data),
    #     "test": Dataset.from_dict(test_data)
    # })

    # # Load tokenizer
    # model_checkpoint = "distilbert-base-uncased"  # Replace with your model
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # # Preprocessing function
    # def preprocess_function(examples):
    #     # Tokenizer expects a list of strings, so we pass the 'text' field directly
    #     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # # breakpoint()
    # # Preprocess the dataset
    # encoded_ds = ds.map(preprocess_function, batched=True)

    # # Set the format for PyTorch
    # encoded_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # # Print some samples to verify correctness
    # print(f"First train sample: {encoded_ds['train'][0]}")
    # print(f"First test sample: {encoded_ds['test'][0]}")
