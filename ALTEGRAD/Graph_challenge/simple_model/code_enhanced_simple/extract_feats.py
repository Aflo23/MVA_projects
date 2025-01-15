import os
from tqdm import tqdm
import random
import re

random.seed(32)


def extract_numbers(text, as_float=True):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    if as_float:
        # Convert to float if needed
        return [float(num) for num in numbers]
    return numbers


# def extract_feats(file):
#     stats = []
#     #print(f'file : {file}')
#     if 
#     fread = open(file,"r")

#     line = fread.read()
#     line = line.strip()
#     stats = extract_numbers(line)
#     fread.close()
#     return stats

def extract_feats(file):
    stats = []
    # Check if the file exists
    if not os.path.exists(file):
        print(f"Warning: File not found: {file}")
        return stats  # Return an empty list if the file doesn't exist

    # Process the file if it exists
    with open(file, "r") as fread:
        line = fread.read().strip()
        stats = extract_numbers(line)
    
    return stats