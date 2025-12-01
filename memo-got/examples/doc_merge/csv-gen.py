import pandas as pd
import argparse
import unicodedata
import sys
import os

def to_ascii(text):
    """
    Converts text to ASCII, handling common smart quotes and special characters
    to prevent data loss before stripping non-ASCII characters.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Manually replace common "smart" characters with ASCII equivalents
    replacements = {
        '‘': "'", '’': "'",
        '“': '"', '”': '"',
        '–': '-', '—': '-',
        '…': '...',
        '\xa0': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Normalize unicode characters (e.g., separating accents from letters)
    text = unicodedata.normalize('NFKD', text)
    
    # Encode to ASCII, ignoring errors (stripping unknown chars), then decode back
    return text.encode('ascii', 'ignore').decode('ascii')

def process_csv(input_file, output_file):
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    print(f"Reading from: {input_file}")

    # Read the input CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Specific columns we expect in the input
    doc_cols = ['document1', 'document2', 'document3', 'document4']
    
    # Check if columns exist
    if not all(col in df.columns for col in doc_cols):
        print(f"Error: Input CSV must contain columns: {doc_cols}")
        sys.exit(1)

    new_rows = []
    chunk_size = 4
    
    # Iterate through the dataframe in steps of 4
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        
        # If we don't have a full chunk of 4, we stop
        if len(chunk) < chunk_size:
            print(f"Warning: Dropping last {len(chunk)} rows (need exactly 4 rows to merge).")
            break

        # Collect all documents from these 4 rows into a single list
        combined_docs = []
        for _, row in chunk.iterrows():
            for col in doc_cols:
                combined_docs.append(row[col])

        # Prepare the new row dictionary
        new_row_data = {
            'id': i // chunk_size, 
            'problem': "Merging sixteen documents with complementary information"
        }

        # Add the 16 documents to the dictionary and clean them
        for idx, doc_text in enumerate(combined_docs):
            col_name = f'document{idx + 1}'
            new_row_data[col_name] = to_ascii(doc_text)

        new_rows.append(new_row_data)

    # Create the new DataFrame
    if not new_rows:
        print("No complete groups of 4 rows found. Output file will be empty.")
        return

    new_df = pd.DataFrame(new_rows)
    
    # Ensure columns are in the correct order: id, problem, doc1..doc16
    cols_order = ['id', 'problem'] + [f'document{n}' for n in range(1, 17)]
    new_df = new_df[cols_order]

    # Save to CSV
    try:
        new_df.to_csv(output_file, index=False)
        print(f"Success! Processed {len(df)} rows into {len(new_df)} rows.")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Merge CSV rows: Takes groups of 4 rows (with 4 docs each) and merges them into 1 row (with 16 docs)."
    )
    
    # Add arguments
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path where the output CSV file will be saved")

    # Parse arguments
    args = parser.parse_args()

    # Run the process
    process_csv(args.input_file, args.output_file)
