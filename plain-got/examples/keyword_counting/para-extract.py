import csv
import argparse
import sys
import re

def split_into_sentences(text):
    """
    Splits text into sentences using a regular expression.
    It looks for sentence terminators (.?!) followed by whitespace 
    or the end of the string.
    """
    if not text:
        return []
    
    # Split by '.', '?', or '!' followed by a space or end of line.
    # The lookbehind (?<=[.!?]) ensures we keep the punctuation with the sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty strings just in case
    return [s for s in sentences if s.strip()]

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(description="Process text from a CSV file.")
    parser.add_argument("csv_file", help="Path to the input CSV file")
    parser.add_argument("n", type=int, help="Number of sentences to extract from the start")
    parser.add_argument("k", type=int, help="Number of times to repeat the extracted block")
    parser.add_argument("output_file", help="Path to the output file")

    args = parser.parse_args()

    # 2. Process the file
    try:
        with open(args.csv_file, mode='r', encoding='utf-8', newline='') as f_in:
            reader = csv.DictReader(f_in)
            
            # Verify the 'Text' column exists
            if 'Text' not in reader.fieldnames:
                print("Error: Input CSV must contain a 'Text' column.")
                sys.exit(1)

            with open(args.output_file, mode='w', encoding='utf-8') as f_out:
                
                rows_processed = 0
                
                for row in reader:
                    # Stop after 20 rows
                    if rows_processed >= 20:
                        break

                    original_text = row.get('Text', '')
                    
                    # 3. Split into sentences
                    all_sentences = split_into_sentences(original_text)
                    
                    # 4. Take the first n sentences
                    # (If text has fewer than n sentences, this takes all of them)
                    selected_sentences = all_sentences[:args.n]
                    
                    if not selected_sentences:
                        # Handle empty rows gracefully
                        f_out.write("\n")
                        rows_processed += 1
                        continue

                    # 5. Concatenate them k times
                    # In Python, multiplying a list repeats its elements
                    repeated_block = selected_sentences * args.k
                    
                    # Join them back into a single paragraph string
                    final_paragraph = ' '.join(repeated_block)
                    
                    # 6. Write to output
                    f_out.write(final_paragraph + "\n\n")
                    
                    rows_processed += 1
                    
        print(f"Successfully processed {rows_processed} rows.")
        print(f"Output written to: {args.output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
