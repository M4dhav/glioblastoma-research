import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from open_dataset_tools.ivy_gap_utils import section_image_loader
from PIL import Image
import io
import torchvision.transforms as transforms

class ISHImageDataset(Dataset):
    def __init__(self, batch_ids, clinical_data_path, section_metadata_path, transform=None):
        self.batch_ids = batch_ids
        self.clinical_data_path = clinical_data_path
        self.section_metadata_path = section_metadata_path
        self.transform = transform

        self.section_metadata = pd.read_csv(self.section_metadata_path)
        self.clinical_data = self.load_clinical_data()
        self.image_data = self.load_images()

    def load_clinical_data(self):
        
        clinical_df = pd.read_csv(self.clinical_data_path)
        clinical_df = clinical_df[clinical_df['donor_id'].isin(self.batch_ids)]
        return clinical_df

    def load_images(self):
        image_data = {}
        for donor_id in self.batch_ids:
            try:
                section_rows = self.section_metadata[self.section_metadata['section_data_set_id'] == donor_id]
                if section_rows.empty:
                    print(f"No section metadata found for donor {donor_id}")
                    continue

                section_image_table = section_image_loader(
                    section_meta_table=section_rows,
                    section_data_set_id=donor_id,
                    verbose=True
                )

                donor_images = []
                for img_promise in section_image_table["primary"]:
                    image = img_promise.load()
                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = transforms.ToTensor()(image)
                    donor_images.append(image)
                image_data[donor_id] = donor_images

            except Exception as e:
                print(f"Error loading images for donor {donor_id}: {e}")

        return image_data

    def __len__(self):
        return len(self.batch_ids)

    def __getitem__(self, idx):
        donor_id = self.batch_ids[idx]
        donor_clinical = self.clinical_data[self.clinical_data['donor_id'] == donor_id].iloc[0]
        survival_days = donor_clinical['survival_days']
        images = self.image_data.get(donor_id, [])

        return images, torch.tensor(survival_days, dtype=torch.float32)
