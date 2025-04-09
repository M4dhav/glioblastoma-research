import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, clinical_input_dim):
        super().__init__()

        # CNN branch
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)

        # Clinical branch
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict survival days
        )

    def forward(self, image, clinical):
        img_feat = self.cnn(image)
        clin_feat = self.clinical_net(clinical)
        x = torch.cat((img_feat, clin_feat), dim=1)
        return self.classifier(x)
