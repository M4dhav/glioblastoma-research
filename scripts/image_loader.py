from torch.utils.data import Dataset
import torchvision.transforms as transforms
from open_dataset_tools.ivy_gap_utils import section_image_loader
import io
from PIL import Image
import torch
import pandas as pd


class ISHImageDataset(Dataset):
    def __init__(self, donor_ids, clinical_df, image_type='primary', transform=None):
        self.donor_ids = donor_ids
        self.clinical_df = clinical_df
        self.image_type = image_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image_data = self.load_images()

    def load_images(self):
        section_meta = pd.read_csv('data/section_metadata.csv')
        section_meta = section_meta[section_meta['donor_id'].isin(self.donor_ids)]

        section_images = section_image_loader(
            section_meta_table=section_meta,
            section_data_set_id=section_meta['section_data_set_id'].tolist(),
            verbose=True
        )

        return [
            (img.load(), row['donor_id']) 
            for img, row in zip(section_images[self.image_type], section_meta.itertuples())
        ]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image, donor_id = self.image_data[idx]
        image = Image.open(io.BytesIO(image.tobytes())).convert("RGB")
        image = self.transform(image)

        clinical_row = self.clinical_df[self.clinical_df['donor_id'] == donor_id].iloc[0]
        clinical_features = torch.tensor(clinical_row.drop('survival_days').values, dtype=torch.float)
        label = torch.tensor(clinical_row['survival_days'], dtype=torch.float)

        return image, clinical_features, label
