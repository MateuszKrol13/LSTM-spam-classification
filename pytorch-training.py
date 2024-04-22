from clean_csv import prepare_input_data, get_vocab_len, build_vocab, tokenize_with_vocab
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import torch

EPOCHS = 1024
BATCH = 64

class EarlyStopper:
    def __init__(self, patience=8, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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
    def __init__(self, vocab_size, embedding_dim):
        super(NeuralNetwork, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, embedding_dim)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(embedding_dim * embedding_dim, 32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.word_embeddings(x)
        x, (final_hidden_state, final_cell_state) = self.lstm1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for example, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        '''
        Use TQDM instead.
        
        if example % 5 == 0:
            loss, current = loss.item(), (example + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        '''

    return loss

# Testing loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss

#detect device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

dataset = prepare_input_data("email_spam.csv")
vocab = build_vocab(dataset)
dataset = tokenize_with_vocab(dataset, vocab)

torch_dataset = SpamDataset(torch.tensor(dataset.loc[:,'vectors']), dataset.loc[:,'class'])
print("The length of the dataset is:", len(torch_dataset))
test_len = int(len(torch_dataset) * 0.3)
train_data, tmp_data = random_split(torch_dataset, [len(torch_dataset) - test_len, test_len])
test_data, validate_data = random_split(tmp_data, [int(test_len / 2), test_len - int(test_len / 2)])
print("The length of train data is:",len(train_data))
print("The length of test data is:",len(test_data))
print("The length of validation data is:",len(validate_data))

# Magic numbers
sentence_length = 70
vocab_len = len(vocab)

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=BATCH)
test_dataloader = DataLoader(test_data, batch_size=BATCH)
validate_dataloader = DataLoader(validate_data, batch_size=BATCH)

model = NeuralNetwork(vocab_len, sentence_length).to(device)
early_stopper = EarlyStopper(patience=3)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss = test(test_dataloader, model, loss_fn)

    print(f"Train Loss: {train_loss:>5f}\nTest Loss: {test_loss:>5f}\n")

    if early_stopper.early_stop(test_loss):
        break

print(f"Validation\n-------------------------------")

test(validate_dataloader, model, loss_fn)

print("Done!")