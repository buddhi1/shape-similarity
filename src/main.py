# main file to call similar shape finding. 

# Author: Buddhi Ashan M. K.
# Date: 07-28-2024

import argparse
import sys
import os

# Add the project root to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules from the project
from data_processing.encode_quad_tree import encode_quad_tree_parks, encode_quad_tree_sports, encode_quad_tree_water_bodies
from data_processing.encode_uniform import encode_uniform_parks, encode_uniform_sports, encode_uniform_water_bodies
from evaluation.groundtruth import ground_truth_parks, ground_truth_sports, ground_truth_water_bodies
from indexing.index_construct import index_parks, index_sports, index_water_bodies
from evaluation.evaluate import evaluate_parks, evaluate_parksth, evaluate_sports, evaluate_water_bodies 


def validate_arguments(args):
    valid_datasets = ['sports', 'water_bodies', 'parks']
    valid_tasks = ['encode_quad_tree', 'encode_uniform', 'groundtruth', 'indexing', 'evaluation']
    
    if args.dataset not in valid_datasets:
        print(f"Error: Invalid dataset name '{args.dataset}'. Valid options are: {', '.join(valid_datasets)}.")
        sys.exit(1)
    
    if args.task not in valid_tasks:
        print(f"Error: Invalid task '{args.task}'. Valid options are: {', '.join(valid_tasks)}.")
        sys.exit(1)
    
    if args.task != 'evaluation':
        if not args.data_file_path or not args.result_dir:
            print(f"Error: 'data_file_path' and 'result_dir' are required for the {args.task} task.")
            sys.exit(1)
    else:
        if not args.groundtruth_dir or not args.index_dir or not args.encoding_dir:
            print("Error: 'groundtruth_dir', 'index_dir', and 'encoding_dir' are required for the evaluation task.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Shape Similarity Project")
    
    parser.add_argument('--dataset', required=True, help="Name of the dataset (sports, water_bodies, or parks)")
    parser.add_argument('--task', required=True, help="Task to perform (encode_quad_tree, encode_uniform, groundtruth, indexing, or evaluation)")
    parser.add_argument('--data_file_path', help="Path to the data file")
    parser.add_argument('--result_dir', help="Directory to save results")
    parser.add_argument('--groundtruth_dir', help="Directory containing ground truth data (for evaluation)")
    parser.add_argument('--index_dir', help="Directory containing index data (for evaluation)")
    parser.add_argument('--encoding_dir', help="Directory containing encoding data (for evaluation)")
    
    args = parser.parse_args()
    
    validate_arguments(args)
        
    # Dynamically construct the function name
    function_name = f"{args.task}_{args.dataset}"
    
    # Get the function from the current module
    try:
        func = globals()[function_name]
    except KeyError:
        print(f"Error: No function '{function_name}' found.")
        sys.exit(1)
    
    print(f"Starting {args.task} task for {args.dataset} dataset...")
    
    # Call the function with the appropriate arguments
    if args.task != 'evaluation':
        func(args.data_file_path, args.result_dir)
    else:
        func(args.groundtruth_dir, args.index_dir, args.encoding_dir)

if __name__ == "__main__":
    main()