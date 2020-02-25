import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data
import model


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
nh = 64
num_epochs = 80
batch_size = 32
learning_rate = 0.001

# MNIST dataset
train_dataset = data.BlinkDataset('./fake_all_frames')

test_dataset = data.BlinkDataset('./blink_crop_frames')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

resnetLSTM = model.ResnetLSTM(nh=nh).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(resnetLSTM.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.float()
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = resnetLSTM(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
resnetLSTM.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnetLSTM(images)
        predicted = outputs.data > 0.5
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(resnetLSTM.state_dict(), 'model.ckpt')