import pandas as pd
import random
import re
from datetime import datetime, timedelta

# Load data
df = pd.read_json("results.json")  

# Remove records where "price" is "Price hidden"
df = df[df["price"] != "Price hidden"]

df = df.drop(columns=["image", "source", "f"], errors="ignore")

# Remove records where "number" is missing/empty
df = df.dropna(subset=["number"])  
df = df[df["number"].astype(str).str.strip() != ""]

# Convert price to numeric (removing "AED " and commas)
df["price"] = df["price"].astype(str).str.replace("AED ", "").str.replace(",", "").astype(int)

# Apply price filter (keep only records where 1,100 ≤ price ≤ 4,000,000)
df = df[(df["price"] >= 1100) & (df["price"] <= 4000000)]

# Keep only Dubai plates
df = df[df["emirate"].str.lower() == "dubai"]

# List of characters to generate for each Dubai plate (A-Z + AA-DD)
characters = [chr(c) for c in range(65, 91)] + ["AA", "BB", "CC", "DD"]  

# Convert "duration" to timestamps
def convert_duration(duration):
    match = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", str(duration))
    if match:
        value, unit = int(match.group(1)), match.group(2)
        if unit == "minute":
            return datetime.now() - timedelta(minutes=value)
        elif unit == "hour":
            return datetime.now() - timedelta(hours=value)
        elif unit == "day":
            return datetime.now() - timedelta(days=value)
        elif unit == "week":
            return datetime.now() - timedelta(weeks=value)
        elif unit == "month":
            return datetime.now() - timedelta(days=value * 30)  # Approximate months as 30 days
        elif unit == "year":
            return datetime.now() - timedelta(days=value * 365)  # Approximate years as 365 days
    return None  # If duration is invalid

# Apply conversion function
df["timestamp"] = df["duration"].apply(convert_duration)

# Remove records with invalid timestamps
df = df.dropna(subset=["timestamp"])

# Separate records where "character" is empty
df_empty_character = df[df["character"].astype(str).str.strip() == ""].copy()
df_non_empty_character = df[df["character"].astype(str).str.strip() != ""]

# Function to generate 4 random variants only
def generate_plate_variants(row):
    selected_chars = random.sample(characters, 4)  
    return [{"number": row["number"], "character": char, "price": row["price"], "emirate": row["emirate"], "timestamp": row["timestamp"]} for char in selected_chars]

# Expand only the records with empty "character"
expanded_records = df_empty_character.apply(generate_plate_variants, axis=1).explode().apply(pd.Series)

# Combine back with the original non-empty records
final_df = pd.concat([df_non_empty_character, expanded_records], ignore_index=True)

final_df = final_df.dropna(subset=["character"]) 

# Keep only the latest record based on duration for each (number, character)
final_df = final_df.sort_values(by="timestamp", ascending=False).drop_duplicates(subset=["number", "character"], keep="first")



# Save cleaned data
final_df.to_csv("cleaned_plates_dubai.csv", index=False)  

print("✅ Data cleaned, filtered, expanded for empty characters, and saved successfully!")
