import os

def rename_files(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    file_list = os.listdir(folder_path)
    file_list.sort()

    # Initialize count for renaming
    count = 1

    # Iterate through files
    for old_name in file_list:
        # Construct new file name with incremental number
        new_name = f"{count}.jpg"
        new_path = os.path.join(folder_path, new_name)

        # Increment count
        count += 1

        # Rename the file
        os.rename(os.path.join(folder_path, old_name), new_path)
        print(f"Renamed {old_name} to {new_name}")

# Example usage
folder_path = "/Users/riccardocasciotti/Downloads/BIS1/photos"
rename_files(folder_path)