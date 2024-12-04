import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImagesDataset
from architecture import MyCNN
from sklearn.model_selection import train_test_split


model = MyCNN()

def train_model(train_loader, model, criterion, optimizer, num_epochs, validation_loader):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels, _, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train


        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for i, data1 in enumerate(validation_loader, 0):
                inputs, labels, _, _ = data1
                outputs = model(inputs)
                loss_validation = criterion(outputs, labels)
                running_loss_val += loss_validation.item()


                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        epoch_loss = running_loss / len(train_loader)
        val_loss = running_loss_val / len(validation_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')


        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "model.pth")

    print('Finished Training')


main_data = ImagesDataset('./training_data')
train_data, valid_data = train_test_split(main_data, test_size=0.1, random_state=42)

print('Training samples:', len(train_data))


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(valid_data, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 35
train_model(train_loader, model, criterion, optimizer, num_epochs, validation_loader)
