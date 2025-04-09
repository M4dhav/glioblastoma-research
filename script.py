import pandas as pd
import pickle
from pathlib import Path

# Load the preprocessed clinical data
clinical_df = pd.read_csv("data/donor_metadata.csv")

# Define batch size and output paths
batch_size = 4
output_dir = Path("data/batches")
output_dir.mkdir(parents=True, exist_ok=True)

# Get all unique donor_ids
donor_ids = clinical_df["donor_id"].unique()
num_donors = len(donor_ids)

# Split donor_ids into batches of 4, last batch may be smaller
donor_batches = [donor_ids[i:i + batch_size] for i in range(0, num_donors, batch_size)]

# Save the batches as a pickle file
with open("data/donor_ids_by_batch.pkl", "wb") as f:
    pickle.dump(donor_batches, f)

# Save separate CSVs for each batch
for i, batch_ids in enumerate(donor_batches):
    try:
        batch_df = clinical_df[clinical_df["donor_id"].isin(batch_ids)]
        if batch_df.empty:
            print(f"⚠️ Batch {i+1} is empty, skipping.")
            continue
        batch_df.to_csv(output_dir / f"clinical_batch_{i+1}.csv", index=False)
        print(f"✅ Saved clinical_batch_{i+1}.csv with {len(batch_df)} rows")
    except Exception as e:
        print(f"❌ Error in batch {i+1}: {str(e)}")

print(f"\n✅ Finished saving {len(donor_batches)} batches.")
