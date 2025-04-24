# main.py

from scripts.image_loader import ISHImageDataset
from scripts.model import MultimodalSurvivalModel
from scripts.train import train_model
import pandas as pd
import pickle

if __name__ == "__main__":
    # Load preprocessed clinical data
    df = pd.read_csv("data/donor_metadata.csv")
    section_df = pd.read_csv("data/section_metadata.csv")
    # Load batch-wise donor IDs
    with open("data/donor_ids_by_batch.pkl", "rb") as f:
        all_batches = pickle.load(f)
        

    for batch_idx, donor_ids in enumerate(all_batches):
        print(f"Training on batch {batch_idx} with {len(donor_ids)} donors...")

        dataset = ISHImageDataset(
            batch_ids=donor_ids,
            # clinical_df=df,
            clinical_data_path= "data/donor_metadata.csv",
            section_metadata_path="data/section_metadata.csv",
            # image_type='primary',  # could also use 'expression', 'annotation', etc.
        )
        print(dataset)
        model = MultimodalSurvivalModel(clinical_input_size=dataset[0][1].shape[0])
        train_model(model, dataset, epochs=10)
