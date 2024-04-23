import argparse
import sys
import torch

def pretty_print(data, indent=0):
    """ Recursively print nested elements. """
    spacing = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}:")
            pretty_print(value, indent + 4)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{spacing} - Element {i}:")
            pretty_print(item, indent + 4)
    elif isinstance(data, torch.Tensor):
        print(f"{spacing}Tensor of shape {data.shape}")
        print(f"{spacing}{data}")
    else:
        print(f"{spacing}{data}")

def read_and_output(file_path, log_file_path=None, keys=None):
    # Load the .pt file
    data = torch.load(file_path)

    # If keys are provided, only keep the data for those keys
    if keys:
        data = {key: data[key] for key in keys if key in data}

    # Decide where to output the data: stdout or log file
    if log_file_path:
        with open(log_file_path, 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            pretty_print(data)
            sys.stdout = original_stdout
        print(f"Data written to {log_file_path}")
    else:
        pretty_print(data)

def main():
    parser = argparse.ArgumentParser(description="Read and print/write a .pt file.")
    parser.add_argument("file_path", type=str, help="Path to the .pt file to read")
    parser.add_argument("--log_file_path", type=str, default=None, help="Optional path to log file for writing output")
    parser.add_argument("--keys", type=str, nargs="+", default=None, help="Optional keys to print")
    parser.add_argument("-raw", action="store_true", help="Print raw data instead of pretty printing")

    args = parser.parse_args()
    if args.raw:
        data = torch.load(args.file_path)
        print(data)
    else: 
        read_and_output(args.file_path, args.log_file_path, args.keys)

if __name__ == "__main__":
    main()
