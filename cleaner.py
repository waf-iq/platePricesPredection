import pandas as pd

# Load data
df = pd.read_json("results.json")  # Change to read_csv() if using CSV

# Remove records where "price" is "Price hidden"
df = df[df["price"] != "Price hidden"]
df = df.drop(columns=["image", "duration", "source","f"], errors="ignore")

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

# Separate records where "character" is empty
df_empty_character = df[df["character"].astype(str).str.strip() == ""].copy()
df_non_empty_character = df[df["character"].astype(str).str.strip() != ""]

# Function to generate copies only for records with empty "character"
def generate_plate_variants(row):
    return [{"number": row["number"], "character": char, "price": row["price"], "emirate": row["emirate"]} for char in characters]

# Expand only the records with empty "character"
expanded_records = df_empty_character.apply(generate_plate_variants, axis=1).explode().apply(pd.Series)

# Combine back with the original non-empty records
final_df = pd.concat([df_non_empty_character, expanded_records], ignore_index=True)

final_df = final_df.dropna(subset=["character"]) 

# Keep only the highest price for each unique (number, character)
final_df = final_df.sort_values(by="price", ascending=False).drop_duplicates(subset=["number", "character"], keep="first")

# Save cleaned data
final_df.to_csv("cleaned_plates_dubai.csv", index=False)  

print("✅ Data cleaned, filtered, expanded for empty characters, and saved successfully!")
