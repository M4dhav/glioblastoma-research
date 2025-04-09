import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalSurvivalModel(nn.Module):
    def __init__(self, clinical_input_size):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        self.fc_clinical = nn.Sequential(
            nn.Linear(clinical_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, clinical):
        image_feat = self.cnn(image)
        clinical_feat = self.fc_clinical(clinical)
        combined = torch.cat((image_feat, clinical_feat), dim=1)
        output = self.combined(combined)
        return output
