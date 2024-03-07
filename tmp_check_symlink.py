

import glob
import os

def check_symlinks(directory):

    assert os.path.exists(directory)

    # Use glob.glob with recursive=True to find all files in directory and subdirectories
    all_files = glob.glob(directory + '/**', recursive=True)
    
    # Iterate through the list of files and check if any is a symlink
    for file in all_files:
        if os.path.islink(file):
            print(f"Symlink found: {file}")
            return True
    
    # If we reach this point, no symlinks were found
    print("No symlinks found.")
    return False

# Replace 'your/folder/path' with the path to the folder you want to check
seeds = range(101, 106) # From 101 to 105
base_directory = "./data/visa_step_random_nr05_tk1_tr60_seed"
# base_directory = "./data/visa_pareto_random_nr05_seed"

symlink_found = False
for seed in seeds:
    directory = f"{base_directory}{seed}"
    if check_symlinks(directory):
        symlink_found = True
        break

if not symlink_found:
    print("No symlinks found in any of the directories.")
else:
    print("Symlink(s) found in one or more of the directories.")