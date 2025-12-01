import csv
import random
import argparse

def generate_lists_csv(filename, num_rows, list_length, unique_count):
    # Configuration
    # The pool of numbers to choose from is 0-9 inclusive
    FULL_RANGE = range(10)
    
    if unique_count > 10 or unique_count < 1:
        raise ValueError("Unique count must be between 1 and 10.")
    
    print(f"Generating {num_rows} rows of data into '{filename}'...")
    print(f"Format: Index, Unsorted List (len={list_length}), Sorted List")
    print(f"Using {unique_count} unique numbers (sampled per row) selected from 0-9")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(["ID", "Unsorted", "Sorted"])

        # Writing rows
        for i in range(num_rows):
            # Pick N unique numbers from 0-9 to be used for this specific row
            selected_digits = sorted(random.sample(FULL_RANGE, unique_count))

            # Generate random list using only the selected digits
            unsorted_list = [random.choice(selected_digits) for _ in range(list_length)]
            
            # Create sorted copy
            sorted_list = sorted(unsorted_list)
            
            # Write to CSV
            writer.writerow([i, str(unsorted_list), str(sorted_list)])

    print("Done!")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate a CSV containing random lists.')
    parser.add_argument('--length', type=int, required=True, help='Length of the lists')
    parser.add_argument('--unique', type=int, required=True, help='Number of unique digits (0-9) to use')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows to generate')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output filename')
    
    args = parser.parse_args()

    generate_lists_csv(args.output, args.rows, args.length, args.unique)
