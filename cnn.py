import torchvision.models as tvm
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from vit import ViTForClassfication as ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from torch.utils.tensorboard.writer import SummaryWriter

config = {
    "patch_size": 32,  # Input image size: 224x224 -> 7x7 patches
    "hidden_size": 768,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 768, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 224,
    "num_classes": 18,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}


'''
Data Processing
'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#Pretrained ViT model weights and transforms
weights_ViT = tvm.ViT_B_32_Weights.IMAGENET1K_V1
weights_transform = weights_ViT.transforms()

#Pretrained CNN ResNet model weights
weights_RsN = tvm.ResNet50_Weights.IMAGENET1K_V1


fullData = datasets.ImageFolder(root="../hagrid-sample-500k-384p/hagrid_500k", transform=transform) # Change tranform to either custom transform or pretrained model
# fullData = datasets.ImageFolder(root="../hagrid-sample-500k-384p/hand_crops_json_processed", transform=transform) # Change tranform to either custom transform or pretrained model
class_names = fullData.classes # number of classes in the dataset
class_names = [name[10:].capitalize() for name in class_names] # Reformat class names and captilize for HAGrid dataset

# ## Create Smaller dataset
samples = 15000 # Update the number of images to use in the dataset
indices = random.sample(range(len(fullData)), samples) 

# Create a subset
fullData = Subset(fullData, indices)

# ##

trainS = int(.8 * len(fullData)) #
valS = int(.1 * len(fullData)) #
testS = len(fullData) - trainS - valS

batch_size = 4 # Update batch size
random_seed = 5 # Update seed

trainData, valData, testData = random_split(fullData, [trainS, valS, testS], generator=torch.Generator().manual_seed(random_seed))

trainLoader = DataLoader(trainData, batch_size = batch_size, shuffle = True)
valLoader = DataLoader(valData, batch_size = batch_size, shuffle= True)
testLoader = DataLoader(testData, batch_size = batch_size)

'''
CNN-ViT Model
'''
class CNNViT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Based on response from:
        # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch

        # Create new CNN model without last two layers

        cnn = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)

        cnnList = []
        for child in cnn.children():
            # Get rid of AdaptiveAvgPool2d and FC layers from resnet
            if child == list(cnn.children())[-2] or child == list(cnn.children())[-1]:
                continue
            cnnList.append(child)

        self.cnn = torch.nn.Sequential(*cnnList)

        self.vitProj = torch.nn.Linear(in_features = 2048, out_features = 768)

        self.vit = ViT(config)


    def forward(self, x):
        # Get Features from cnn
        # Initial shape: batch_size, 3, 224, 224
        x = self.cnn(x)
        # Becomes: batch_size, 2048, 7, 7

        # Flatten to make tensor shape 128, 7*7, 2048
        x = x.reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
        # Becomes: batch_size, 49, 2048

        # Project to ViT dimension
        x = self.vitProj(x)
        # Becomes: batch_size, 49, 768

        x = self.vit(x)

        return x[0]

model = CNNViT()

## Option to load existing model. Comment out as needed
# state_dict = torch.load("./ckpt/cnnvit2.pth") # Update file name 
# model.load_state_dict(state_dict)

device = torch.device("cuda")

model.to(device)
model.vit.to(device)
model.cnn.to(device)


'''
Standalone CNN ResNet Model
'''

# Load pretrained ResNet50
model_RsN = tvm.resnet50(weights=weights_RsN)

# # Freeze all the pretrained layers
# for param in model_RsN.parameters():
#     param.requires_grad = False

# Replace final fully connected layer
num_ftrs = model_RsN.fc.in_features
model_RsN.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Number of output classes
model_RsN = model_RsN.to(device)

# # Only the final layer's parameters will be updated
# for param in model_RsN.fc.parameters():
#     param.requires_grad = True


'''
Standalone ViT Model
'''

## ViT Only Model
# model_vit = ViT(config)
# model_vit = model_vit.to(device)


# Pre-trained Pytorch ViT
model_pyT_vit = tvm.vit_b_32(weights=weights_ViT)


for params in model_pyT_vit.parameters():
    params.requires_grad=False

# UNFREEZE last encoder blocks
# ViT encoder blocks are in model.encoder.layers
LayerTrStart = 6
LayerTrEnd = 12
for i in range(LayerTrStart, LayerTrEnd):  # Last 3 blocks (index 9, 10, 11)
    for param in model_pyT_vit.encoder.layers[i].parameters():
        param.requires_grad = True


#Changing Last Layer
model_pyT_vit.heads.head = torch.nn.Linear(model_pyT_vit.heads.head.in_features, len(class_names))

for param in model_pyT_vit.heads.parameters():
    param.requires_grad = True

model_pyT_vit = model_pyT_vit.to(device)


'''
Training and Test Functions
'''
# Training function
def train_model(model, m_name, lossFN, optimizer, num_epochs=10):
    since = time.time()

    print("-" * 10)

    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []
    # Learning rate scheduler setup
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(num_epochs):
        
        for phase in ["train", "val"]:
        # for phase in ["train"]:
            if phase == "train":
                model.train()
                loader = trainLoader
            else:
                model.eval()
                loader = valLoader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = lossFN(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            if phase == "train":    
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
                print(f"Epoch {epoch+1}/{num_epochs}: {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}") # prints only train

            if phase == "val": 
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())

                # Step the scheduler with validation loss
                scheduler.step(epoch_loss)

            # print(f"Epoch {epoch+1}/{num_epochs}: {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}") # prints both train and val set

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_loss, color='blue', label="Train")
    plt.plot(range(1, num_epochs+1), val_loss, color='green', label="TestVal")
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_acc, color='blue', label="Train")
    plt.plot(range(1, num_epochs+1), val_acc, color='green', label="TestVal")
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f"{m_name}, Training Time:{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s", fontsize=12)

    if m_name == "Pretrained_ViT":
        plt.savefig(f"./{checkpointPath}/{m_name}_{samples}samples_LR_{lr:.0e}_LayerTr{LayerTrStart}_{LayerTrEnd}_TrainingPlot")
    else: plt.savefig(f"./{checkpointPath}/{m_name}_{samples}samples_LR_{lr:.0e}_TrainingPlot")
    plt.close()

    return train_loss, train_acc

# Testing function
def test_model(model, testLoader,m_name):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")

        # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # print("Confusion matrix:\n", cm)

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Test Accuracy {accuracy:.2f}%')
    plt.tight_layout()
    plt.savefig(f"./{checkpointPath}/{m_name}_{samples}samples_confusion_matrix.png")
    plt.close()


'''
Run the Models
'''
## Run the Models

Models = {
    # "CNN_only": model_RsN,
    # "Fusion_CNN-ViT": model,
    # "ViT_only": model_vit,
    "Pretrained_ViT": model_pyT_vit
}

model_names = list(Models.keys()) # Get model names from dict

# Define loss function
lossFN = torch.nn.CrossEntropyLoss()
lr = 0.001


# Local for saving model
checkpointPath = "ckpt" # File name
os.makedirs(checkpointPath, exist_ok= True) # Create file


# Run through all the models
for name in model_names:

    print(name)
    print(f"Sample size: {samples}")

    if name == "Pretrained_ViT":
        optimizer = torch.optim.AdamW(Models[name].parameters(), lr=lr, weight_decay=0.1) #Weight decay optimizer

    else: optimizer = torch.optim.Adam(Models[name].parameters(), lr=lr) #.0001, .001, .3
    train_loss, train_acc = train_model(Models[name], name, lossFN, optimizer, num_epochs=10)
    torch.save(Models[name].state_dict(), f"{checkpointPath}/{name}_{samples}samples.pth")

    test_model(Models[name], testLoader, name)
    print("\n----------")


## Run Models Manually

# print("CNN ResNet Only Model")
# optimizer = torch.optim.Adam(model_RsN.parameters(), lr=.0001) #.0001, .001, .3
# CNN_only = train_model(model_RsN, lossFN, optimizer, num_epochs=5)
# test_model(model_RsN, testLoader)

# print("Fusion CNN-ViT Model")
# optimizer = torch.optim.Adam(model.parameters(), lr=.0001) #.0001, .001, .3
# Fusion = train_model(model, lossFN, optimizer, num_epochs=5)
# test_model(model, testLoader)

# print("ViT Model")
# optimizer = torch.optim.Adam(model_vit.parameters(), lr=.0001) #.0001, .001, .3
# Fusion = train_model(model_vit, lossFN, optimizer, num_epochs=5)
# test_model(model_vit, testLoader)

# print("Pre-Train PyTorch ViT Model")
# optimizer = torch.optim.Adam(model_pyT_vit.parameters(), lr=.0001) #.0001, .001, .3
# Fusion = train_model(model_pyT_vit, lossFN, optimizer, num_epochs=5)
# test_model(model_pyT_vit, testLoader)