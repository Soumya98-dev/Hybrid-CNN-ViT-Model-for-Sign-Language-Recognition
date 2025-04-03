import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ASLDataset
from multimodal_asl_model import MultiModalAttentionModel
from train_eval import train, evaluate, plot_metrics, print_metrics

def run_training(train_loader, val_loader, test_loader, epochs=10, lr=1e-4, device='cuda'):
    model = MultiModalAttentionModel(num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            best_val_acc = val_acc

    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Final test eval
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print_metrics(y_true, y_pred)

if __name__ == "__main__":
    # You need to load these properly
    train_imgs, train_landmarks, train_labels = ...
    val_imgs, val_landmarks, val_labels = ...
    test_imgs, test_landmarks, test_labels = ...

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = ASLDataset(train_imgs, train_landmarks, train_labels, transform)
    val_dataset = ASLDataset(val_imgs, val_landmarks, val_labels, transform)
    test_dataset = ASLDataset(test_imgs, test_landmarks, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    run_training(train_loader, val_loader, test_loader, epochs=20)
