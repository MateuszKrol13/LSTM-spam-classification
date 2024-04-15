from clean_csv import prepare_input_data
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import torch

EPOCHS = 1024
BATCH = 64

# Define dataset
class SpamDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = X_data
        self.y = y_data

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(96, 96)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(96, 32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X, y

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


dataset = prepare_input_data("email_spam.csv")
torch_dataset = SpamDataset(dataset.loc[:,'vector'], dataset.loc[:,'class'])
print("The length of the dataset is:", len(torch_dataset))
train_len = int(len(torch_dataset) * 2 / 3)
train_data, test_data = random_split(torch_dataset, [train_len, len(torch_dataset) - train_len])
print("The length of train data is:",len(train_data))
print("The length of test data is:",len(test_data))

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=BATCH)
test_dataloader = DataLoader(test_data, batch_size=BATCH)

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")