import re
import csv
import sys
import argparse

# ---------------------------------------------------------
# A truncated list of countries for demonstration.
# In a real scenario, you might read this from a text file
# or copy-paste the full ISO list here.
# ---------------------------------------------------------
COUNTRIES = [
"Afghanistan",
"Albania",
"Algeria",
"Andorra",
"Angola",
"Antigua and Barbuda",
"Argentina",
"Armenia",
"Australia",
"Austria",
"Azerbaijan",
"Bahamas",
"Bahrain",
"Bangladesh",
"Barbados",
"Belarus",
"Belgium",
"Belize",
"Benin",
"Bhutan",
"Bolivia",
"Bosnia and Herzegovina",
"Botswana",
"Brazil",
"Brunei",
"Bulgaria",
"Burkina Faso",
"Burundi",
"Cabo Verde",
"Cambodia",
"Cameroon",
"Canada",
"Central African Republic",
"Chad",
"Chile",
"China",
"Colombia",
"Comoros",
"Congo, Democratic Republic of the",
"Congo, Republic of the",
"Costa Rica",
"Côte d'Ivoire",
"Croatia",
"Cuba",
"Cyprus",
"Czech Republic",
"Denmark",
"Djibouti",
"Dominica",
"Dominican Republic",
"Ecuador",
"Egypt",
"El Salvador",
"Equatorial Guinea",
"Eritrea",
"Estonia",
"Eswatini",
"Ethiopia",
"Fiji",
"Finland",
"France",
"Gabon",
"Gambia",
"Georgia",
"Germany",
"Ghana",
"Greece",
"Grenada",
"Guatemala",
"Guinea",
"Guinea-Bissau",
"Guyana",
"Haiti",
"Honduras",
"Hungary",
"Iceland",
"India",
"Indonesia",
"Iran",
"Iraq",
"Ireland",
"Israel",
"Italy",
"Jamaica",
"Japan",
"Jordan",
"Kazakhstan",
"Kenya",
"Kiribati",
"Kosovo",
"Kuwait",
"Kyrgyzstan",
"Laos",
"Latvia",
"Lebanon",
"Lesotho",
"Liberia",
"Libya",
"Liechtenstein",
"Lithuania",
"Luxembourg",
"Madagascar",
"Malawi",
"Malaysia",
"Maldives",
"Mali",
"Malta",
"Marshall Islands",
"Mauritania",
"Mauritius",
"Mexico",
"Micronesia",
"Moldova",
"Monaco",
"Mongolia",
"Montenegro",
"Morocco",
"Mozambique",
"Myanmar",
"Namibia",
"Nauru",
"Nepal",
"Netherlands",
"New Zealand",
"Nicaragua",
"Niger",
"Nigeria",
"North Korea",
"North Macedonia",
"Norway",
"Oman",
"Pakistan",
"Palau",
"Panama",
"Papua New Guinea",
"Paraguay",
"Peru",
"Philippines",
"Poland",
"Portugal",
"Qatar",
"Romania",
"Russia",
"Rwanda",
"Saint Kitts and Nevis",
"Saint Lucia",
"Saint Vincent and the Grenadines",
"Samoa",
"San Marino",
"São Tomé and Príncipe",
"Saudi Arabia",
"Senegal",
"Serbia",
"Seychelles",
"Sierra Leone",
"Singapore",
"Slovakia",
"Slovenia",
"Solomon Islands",
"Somalia",
"South Africa",
"South Korea",
"South Sudan",
"Spain",
"Sri Lanka",
"Sudan",
"Suriname",
"Sweden",
"Switzerland",
"Syria",
"Tajikistan",
"Tanzania",
"Thailand",
"Timor-Leste",
"Togo",
"Tonga",
"Trinidad and Tobago",
"Tunisia",
"Turkey",
"Turkmenistan",
"Tuvalu",
"Uganda",
"Ukraine",
"United Arab Emirates",
"United Kingdom",
"United States",
"Uruguay",
"Uzbekistan",
"Vanuatu",
"Vatican City",
"Venezuela",
"Vietnam",
"Yemen",
"Zambia",
"Zimbabwe"
]

def get_country_regex(country_list):
    # Sort by length descending to match longest names first
    sorted_countries = sorted(list(set(country_list)), key=len, reverse=True)
    # Escape regex special characters
    escaped = [re.escape(c) for c in sorted_countries]

    # Compile Regex: \b ensures whole word matches
    pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)

def simple_sentence_count(text):
    # Count sentences by splitting on punctuation tokens
    count = len(re.split(r'[.!?]+', text)) - 1
    return max(1, count) if len(text) > 0 else 0

def process_paragraphs(file_path, output_path):
    country_pattern = get_country_regex(COUNTRIES)

    try:
        # Read as ASCII
        with open(file_path, 'r', encoding='ascii', errors='replace') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    # Split content into paragraphs (blank lines separation)
    # We strip whitespace and ignore empty paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    # Output setup
    out_file = open(output_path, 'w', newline='', encoding='ascii', errors='replace') if output_path else sys.stdout

    headers = ['ID', 'Text', 'Countries', 'Sentences', 'Characters']

    try:
        writer = csv.writer(out_file)
        writer.writerow(headers)

        # Use enumerate to get the index (i) starting at 0
        for i, paragraph in enumerate(paragraphs):

            # Find countries
            countries_found = country_pattern.findall(paragraph)

            # Count sentences
            s_count = simple_sentence_count(paragraph)

            # Count characters
            c_count = len(paragraph)

            writer.writerow([
                i,                      # Paragraph Number (0-based)
                paragraph,
                str(countries_found),
                s_count,
                c_count
            ])

    finally:
        if output_path:
            out_file.close()
            print(f"Successfully wrote results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract countries from text (ASCII Mode)")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("-o", "--output", help="Path to output CSV file")

    args = parser.parse_args()

    process_paragraphs(args.input_file, args.output)
