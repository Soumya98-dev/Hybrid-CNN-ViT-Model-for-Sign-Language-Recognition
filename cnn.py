import torchvision.models as tvm
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import random
import time

from vit import ViTForClassfication as ViT

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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

fullData = datasets.ImageFolder(root="../hagrid-sample-500k-384p/hagrid_500k", transform=transform)
class_names = fullData.classes

# ## Create Smaller dataset

indices = random.sample(range(len(fullData)), 500) # Update the number of images to use in the dataset

# Create a subset
fullData = Subset(fullData, indices)

# ##

trainS = int(.8 * len(fullData)) #
valS = int(.1 * len(fullData)) #
testS = len(fullData) - trainS - valS

batch_size = 4

trainData, valData, testData = random_split(fullData, [trainS, valS, testS])

trainLoader = DataLoader(trainData, batch_size = batch_size, shuffle = True)
valLoader = DataLoader(valData, batch_size = batch_size, shuffle= True)
testLoader = DataLoader(testData, batch_size = batch_size)

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

        self.vit = ViT(config)

        self.vitProj = torch.nn.Linear(in_features = 2048, out_features = 768)


    def forward(self, x):
        # Get Features from cnn
        # Initial shape: 128, 3, 224, 224
        x = self.cnn(x)
        # Becomes: 128, 2048, 7, 7

        # Flatten feature map and project to ViT dimension

        # Flatten to make tensor shape 128, 7*7, 2048
        x = x.reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
        # Becomes: 128, 49, 2048

        x = self.vitProj(x)
        # Becomes: 128, 49, 768

        x = self.vit(x)

        return x[0]


model = CNNViT()

checkpointPath = "ckpt"
os.makedirs(checkpointPath, exist_ok= True)

lossFN = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001) #.0001, .001, .3

device = torch.device("cuda")

model.to(device)
model.vit.to(device)
model.cnn.to(device)

# epochs = 5
# print("Fusion Model Training")
# for epoch in range(epochs):
#     model.train()
#     # model.cnn.train()
#     # model.vit.train()
#     for images,labels in trainLoader:
#         images, labels = images.to(device), labels.to(device)

#         yPred = model(images)
#         loss = lossFN(yPred, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

    
#     print(f"Epoch {epoch + 1}, Loss: {loss.item():.5f}")

# #torch.save(model.state_dict(), f"{checkpointPath}/.pth")

# model.eval()
# model.cnn.eval()
# model.vit.eval()
# tc = 0
# tp = 0
# with torch.no_grad():
#     for i, (images, labels) in enumerate(valLoader):
#         images, labels = images.to(device), labels.to(device)

#         # Forwards
#         yPred = model(images)
#         loss = lossFN(yPred, labels)
#         _, predicted = torch.max(yPred, dim = 1)

#         for guess in range(len(predicted)):
#             if predicted[guess] == labels[guess]:
#                 tc += 1
#             tp += 1

#     testAcc = tc / tp
#     print("Fusion Model TEST ACCURACY: " + str(testAcc))



'''
Standalone CNN ResNet Model
'''
# Load pretrained ResNet50
model_RsN = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)

# Replace final fully connected layer
num_ftrs = model_RsN.fc.in_features
model_RsN.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Number of output classes
model_RsN = model_RsN.to(device)


# Training function
def train_model(model, lossFN, optimizer, num_epochs=10):
    since = time.time()

    print("-" * 10)
    
    for epoch in range(num_epochs):
        
        # for phase in ["train", "val"]:
        for phase in ["train"]:
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

            print(f"Epoch {epoch+1}/{num_epochs}: {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return model


def test_model(model, testLoader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")


## Run the Models
CNN_only = train_model(model_RsN, lossFN, optimizer, num_epochs=10)
test_model(model_RsN, testLoader)

Fusion = train_model(model, lossFN, optimizer, num_epochs=10)
test_model(model, testLoader)