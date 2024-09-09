import argparse
import json
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with DeepSpeed config")

    # General arguments
    parser.add_argument('--model_checkpoint', type=str, default="Qwen/Qwen2-1.5B", help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save model and outputs')
    parser.add_argument('--train_dataset_path', type=str, default="./train.json", help='Path to the training dataset')
    parser.add_argument('--test_dataset_path', type=str, default="./test.json", help='Path to the test dataset')
    parser.add_argument('--eval_save_strategy', type=str, default="steps", help='Evaluation and save strategy')
    parser.add_argument('--save_steps', type=int, default=500, help='Steps to save checkpoint')
    parser.add_argument('--eval_steps', type=int, default=500, help='Steps to evaluate the model')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--deepspeed', type=str, default="./dz_zero3.json", help='Path to DeepSpeed config file')
    parser.add_argument('--bf16', type=bool, default=False, help='Whether to use bf16 precision')
    parser.add_argument('--data_type', type=int, default=0, help='Data type for the task')
    parser.add_argument('--use_param_search', action='store_true', help='Enable hyperparameter search')
    parser.add_argument('--sub_name', type=str, default="test", help='Name of the submission')
    return parser.parse_args()

def load_deepspeed_config(deepspeed_config_path):
    with open(deepspeed_config_path, 'r') as file:
        deepspeed_config = json.load(file)
    return deepspeed_config

if __name__ == "__main__":
    args = parse_args()

    # Load the DeepSpeed configuration file
    deepspeed_config = load_deepspeed_config(args.deepspeed)

    # Attach DeepSpeed config as an argument (as string path or as a dictionary, depends on train.py logic)
    args.deepspeed = deepspeed_config

    # Call the train function with parsed arguments
    train(args)
