# python src/main.py --dataset sports --task encode_quad_tree --data_file data/sports --result_dir deleteTesting
# python src/main.py --dataset sports --task encode_quad_tree --data_file data/water_bodies --result_dir deleteTesting
# python src/main.py --dataset sports --task encode_quad_tree --data_file data/parks --result_dir deleteTesting

# python src/main.py --dataset sports --task encode_uniform --data_file data/sports --result_dir deleteTesting
# python src/main.py --dataset sports --task encode_uniform --data_file data/water_bodies --result_dir deleteTesting
# python src/main.py --dataset sports --task encode_uniform --data_file data/parks --result_dir deleteTesting

# python src/main.py --dataset sports --task groundtruth --data_file data/sports --result_dir deleteTesting
# python src/main.py --dataset sports --task groundtruth --data_file data/water_bodies --result_dir deleteTesting
# python src/main.py --dataset sports --task groundtruth --data_file data/parks --result_dir deleteTesting

# python src/main.py --dataset sports --task indexing --data_file data/sports --result_dir deleteTesting
# python src/main.py --dataset sports --task indexing --data_file data/water_bodies --result_dir deleteTesting
# python src/main.py --dataset sports --task indexing --data_file data/parks --result_dir deleteTesting

# python src/main.py --dataset sports --task evaluation --data_file data/sports --result_dir deleteTesting
# python src/main.py --dataset sports --task evaluation --data_file data/water_bodies --result_dir deleteTesting
# python src/main.py --dataset sports --task evaluation --data_file data/parks --result_dir deleteTesting

#!/bin/bash

# Global variables for paths
DATA_DIR="data"
RESULT_DIR="deleteTesting"
GROUNDTRUTH_DIR="deleteTesting/groundtruth"
INDEX_DIR="deleteTesting/index"
ENCODING_DIR="deleteTesting/encoding"

# Usage function to display help
usage() {
  echo "Usage: $0 --task [encode_quad_tree|encode_uniform|groundtruth|indexing|evaluation]"
  echo ""
  echo "This script runs shape similarity tasks for multiple datasets."
  echo ""
  echo "Options:"
  echo "  --task            The task to perform. Valid options are:"
  echo "                    - encode_quad_tree"
  echo "                    - encode_uniform"
  echo "                    - groundtruth"
  echo "                    - indexing"
  echo "                    - evaluation"
  echo ""
  echo "Examples:"
  echo "  $0 --task encode_quad_tree"
  echo "  $0 --task groundtruth"
  echo ""
  echo "Note: The script will run the specified task for all datasets (sports, water_bodies, parks)."
  exit 1
}

# Parse command line arguments
if [ "$#" -eq 0 ]; then
  echo "Error: No arguments provided."
  usage
fi

while [ "$#" -gt 0 ]; do
  case $1 in
    --task) TASK="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Error: Invalid argument $1"; usage ;;
  esac
  shift
done

# Validate arguments
if [ -z "$TASK" ]; then
  echo "Error: --task parameter is required."
  usage
fi

# Function to run the appropriate commands for all datasets
run_task_for_all_datasets() {
  task="$1"
  datasets="sports water_bodies parks"

  for dataset in $datasets; do
    data_file_path="$DATA_DIR/$dataset"

    if [ "$task" = "encode_quad_tree" ]; then
      python src/main.py --dataset "$dataset" --task encode_quad_tree --data_file "$data_file_path" --result_dir "$RESULT_DIR"
    elif [ "$task" = "encode_uniform" ]; then
      python src/main.py --dataset "$dataset" --task encode_uniform --data_file "$data_file_path" --result_dir "$RESULT_DIR"
    elif [ "$task" = "groundtruth" ]; then
      python src/main.py --dataset "$dataset" --task groundtruth --data_file "$data_file_path" --result_dir "$RESULT_DIR"
    elif [ "$task" = "indexing" ]; then
      python src/main.py --dataset "$dataset" --task indexing --data_file "$data_file_path" --result_dir "$RESULT_DIR"
    elif [ "$task" = "evaluation" ]; then
      python src/main.py --dataset "$dataset" --task evaluation --groundtruth_dir "$GROUNDTRUTH_DIR" --index_dir "$INDEX_DIR" --encoding_dir "$ENCODING_DIR"
    else
      echo "Error: Invalid task '$task'."
      usage
    fi
  done
}

# Run the task for all datasets
run_task_for_all_datasets "$TASK"
