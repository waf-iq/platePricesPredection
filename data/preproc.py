import json
import pandas as pd
from collections import Counter
import re
from datetime import datetime, timedelta

def get_most_frequent_digits(number):
    counter = Counter(str(number))
    
    # Sort by frequency (desc), then by digit value (asc)
    sorted_counts = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    # Find the first non-zero digit for X
    value_of_X, number_of_X = None, 0
    for digit, count in sorted_counts:
        if digit != "0":  # Skip 0
            value_of_X, number_of_X = digit, count
            break
    
    # Find the next non-zero digit for Y
    value_of_Y, number_of_Y = None, 0
    for digit, count in sorted_counts:
        if digit != "0" and digit != value_of_X:  # Skip 0 and X
            value_of_Y, number_of_Y = digit, count
            break

    return value_of_X, number_of_X, value_of_Y, number_of_Y

def detect_pattern(number, X, Y, number_of_X, number_of_Y):
    """Detects patterns in plate numbers while keeping zeros intact."""
    X = str(X) if X is not None else ''
    Y = str(Y) if Y is not None else ''

    def replace_digit(match):
        digit = match.group()
        if digit == "0":
            return "0"  # Keep zeros
        elif digit == X:
            return "X"
        elif digit == Y:
            return "Y"
        return "_"  # Replace all other digits with "_"

    pattern = re.sub(r"\d", replace_digit, number)

    if number_of_X == 1:
        pattern = pattern.replace('X', '_')

    if number_of_Y == 1:
        pattern = pattern.replace('Y', '_')

    return pattern

def count_zeros(number):
    return number.count("0")

def is_year_like(number):
    return len(number) == 4 and 1950 <= int(number) <= 2030

def IsPalindrome(number):
    return number == number[::-1]

def IsMemeNumber(number):
    return number in {"69", "420", "666", "911"}

def ContainsMemeNumber(number):
    return any(meme in number for meme in {"69", "420", "666", "911"})

def IsCarModel(number, character):
    return "63" in number and character.upper() in {"G", "C", "E", "S", "A"}

# New function: Check if the number contains a sequence (123, 987, etc.)
def contains_series(number):
    number = str(number)
    for i in range(len(number) - 2):
        segment = number[i:i+3]
        if segment in "0123456789" or segment in "9876543210":
            return True
    return False

# New function: Check if the whole number is a sequential series
def is_full_series(number):
    return number in "0123456789" or number in "9876543210"

# Load CSV into pandas
df = pd.read_csv("cleaned_plates_dubai.csv")

# Convert "number" to string to avoid errors
df["number"] = df["number"].astype(str)

# Process each plate
df["value_of_X"], df["number_of_X"], df["value_of_Y"], df["number_of_Y"] = zip(*df["number"].apply(get_most_frequent_digits))
df["pattern"] = df.apply(lambda row: detect_pattern(row["number"], row["value_of_X"], row["value_of_Y"], row["number_of_X"], row["number_of_Y"]), axis=1)
df["number_of_zeros"] = df["number"].apply(count_zeros)
df["number_of_digits"] = df["number"].apply(len)
df["isYear"] = df["number"].apply(is_year_like)
df["isPalindrome"] = df["number"].apply(IsPalindrome)
df["isMemeNumber"] = df["number"].apply(IsMemeNumber)
df["containsMemeNumber"] = df["number"].apply(ContainsMemeNumber)
df["isCarModel"] = df.apply(lambda row: IsCarModel(row["number"], row["character"]), axis=1)
df["containsSeries"] = df["number"].apply(contains_series)
df["isFullSeries"] = df["number"].apply(is_full_series)

# Save as JSON
df.to_json("enhanced_plates.json", orient="records", indent=4)

# Save as CSV
df.to_csv("enhanced_plates_data.csv", index=False)

print("✅ Processing complete. Data saved to enhanced_plates.json and enhanced_plates_data.csv.")
