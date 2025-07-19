import pandas as pd
import re
import os
import json

# Configuration
INPUT_FILE = "data/raw_data_20250718_002943.csv"  # Update to your latest file
OUTPUT_CSV = "data/cleaned_data.csv"
OUTPUT_TXT = "data/loan_data.txt"
OUTPUT_JSON = "data/cleaned_data.json"  # Required for embeddings pipeline

def clean_text(text):
    """Enhanced text cleaning function"""
    if not isinstance(text, str):
        return ""
    
    # Normalize whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve important financial symbols and units
    text = re.sub(r'[^\w\s₹$%.,-]', '', text)
    
    # Remove sequences of special characters
    text = re.sub(r'([.,-])\1+', r'\1', text)
    
    return text.strip()

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load raw data
    print(f"📖 Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Clean content
    print("🧹 Cleaning text content...")
    df["cleaned"] = df["content"].apply(clean_text)
    
    # Filter out empty content
    df = df[df["cleaned"].str.len() > 50]  # Remove very short entries
    
    # Save to CSV (original format)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 CSV saved to {OUTPUT_CSV}")
    
    # Save to TXT (one document per loan type)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for url, content in zip(df["url"], df["cleaned"]):
            f.write(f"URL: {url}\n")
            f.write(f"CONTENT:\n{content}\n\n")
    print(f"📄 TXT saved to {OUTPUT_TXT}")
    
    # Save to JSON (for embeddings pipeline)
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            "url": row["url"],
            "content": row["cleaned"],
            "type": row["url"].split("/")[-1].replace("-loan", "").title() + " Loan"
        })
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"🔮 JSON saved to {OUTPUT_JSON}")
    
    # Print summary
    print("\n✅ Cleaning complete! Results:")
    print(f"- Total entries: {len(df)}")
    print(f"- Avg length: {df['cleaned'].str.len().mean():.0f} chars")
    print(f"- Total chars: {df['cleaned'].str.len().sum():,}")

if __name__ == "__main__":
    main()