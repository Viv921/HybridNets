import os
import glob

# --- 1. CONFIGURE THIS SECTION ---

# IMPORTANT: Change this to the full path of your folder containing the .txt files
folder_path = "../../datasets_ta/leftimg/val/v_label"

# The class IDs you are looking for
missing_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8}

# ---------------------------------

# This dictionary will store our results
# Key: class_id, Value: a set of filenames
found_in_files = {class_id: set() for class_id in missing_classes}

# Create the search pattern for glob
search_pattern = os.path.join(folder_path, "*.txt")

print(f"Scanning for .txt files in: {folder_path}\n")

# Loop through all files matching the pattern
for file_path in glob.glob(search_pattern):
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Get the class ID (the first number on the line)
                try:
                    class_id = int(line.split()[0])
                    
                    # If this is one of the classes we're looking for,
                    # add the filename to our results
                    if class_id in missing_classes:
                        found_in_files[class_id].add(filename)
                        
                except (ValueError, IndexError):
                    # Handle lines that are empty or don't start with a number
                    print(f"Skipping bad line in {filename}: {line.strip()}")
                    
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# --- 3. Print the Results ---
print("\n" + "---" * 10)
print("      Scan Complete: Results")
print("---" * 10 + "\n")

total_found_count = 0
for class_id, filenames in found_in_files.items():
    if filenames:
        total_found_count += 1
        print(f"✅ Found Class ID: {class_id}")
        print(f"Present in the following {len(filenames)} file(s):")
        # Sort the filenames for a clean, alphabetical list
        for filename in sorted(list(filenames)):
            print(f"  - {filename}")
        print("-" * 20)
    else:
        print(f"❌ Class ID {class_id} was NOT found in any files.")

if total_found_count == 0:
    print("\nUnfortunately, none of the missing classes were found in any files.")