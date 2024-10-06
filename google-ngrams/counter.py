import csv
import glob

def get_total_ngrams_for_year(directory, year):
    # Initialize a dictionary to store counts for each n-gram type (1-gram, 2-gram, etc.)
    ngram_totals = {}

    # Iterate through different n-gram files (totalcounts-1, totalcounts-2, etc.)
    for n in range(1, 6):  # You can adjust the range based on the number of n-grams (1-gram to 5-grams)
        file_pattern = f'{directory}/totalcounts-{n}'  # Match files like totalcounts-1, totalcounts-2, etc.
        ngram_files = glob.glob(file_pattern)

        # Initialize the count for this n-gram type
        ngram_totals[f'{n}-grams'] = 0

        for file_path in ngram_files:
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')  # Assuming tab-delimited, adjust if necessary
                for row in reader:
                    for entry in row[1:]:  # Skip the ngram part, focus on year, match_count triplets
                        # Each entry is expected to be in the form "year,match_count,page_count,volume_count"
                        try:
                            year_data = entry.split(',')
                            ngram_year = int(year_data[0])
                            match_count = int(year_data[1])

                            # Check if the year matches the specified year
                            if ngram_year == year:
                                ngram_totals[f'{n}-grams'] += match_count
                        except (ValueError, IndexError):
                            # Skip the line if there's an issue with parsing
                            continue

    return ngram_totals

# Directory where the ngram files are located
directory = '.'

# Specify the year we want to sum up (e.g., 2019)
year = 2019

# Get the total ngram counts for the year 2019
ngram_counts = get_total_ngrams_for_year(directory, year)

# Display the results
for ngram_type, count in ngram_counts.items():
    print(f'Total number of {ngram_type} in {year}: {count}')

