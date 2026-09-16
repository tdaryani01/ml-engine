import csv
import os

MALICIOUS_PATH = ".\\data\\samples\\csv\\github_ingested_seeds.csv"
BENIGN_PATH = ".\\data\\samples\\csv\\benign_control_seeds.csv"
OUTPUT_PATH = ".\\data\\samples\\csv\\final_balanced_dataset.csv"

def merge_datasets():
    if not os.path.exists(MALICIOUS_PATH) or not os.path.exists(BENIGN_PATH):
        print("[ERROR] Missing input files. Make sure both CSV files exist!")
        return

    merged_rows = []
    
    # Load Malicious Payloads (Targets 1 & 2)
    with open(MALICIOUS_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            merged_rows.append(row)

    # Load Benign Controls (Target 0)
    with open(BENIGN_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            merged_rows.append(row)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = ["seed_id", "category_type", "subcategory", "prompt_text", "target"]
    
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    unique_seeds = len(set(r["seed_id"] for r in merged_rows))
    print(f"\n[SUCCESS] Master balanced dataset created at '{OUTPUT_PATH}'")
    print(f"Total Rows: {len(merged_rows)}")
    print(f"Total Unique Seed Groups: {unique_seeds}")

if __name__ == "__main__":
    merge_datasets()