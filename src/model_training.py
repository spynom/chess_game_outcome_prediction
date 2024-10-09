import numpy as np
import torch
import json
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("cuda is available" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

class ThreeLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=8,dropout=0.0):
        super(ThreeLayerLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward propagate through LSTM
        lstm_out, _ = self.lstm(x)

        # Get the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        self.last_time_step = last_time_step

        output=self.fc(self.last_time_step)

        return output


def accuracy_score_check(y,y_hat):
    y=y.cpu().numpy().reshape(-1)
    try:
        y_hat=y_hat.cpu().reshape(-1,3)
        _,y_hat=torch.max(y_hat,dim=1)
    except RuntimeError as e:
        y_hat = y_hat.detach().cpu().reshape(-1, 3)
        _, y_hat = torch.max(y_hat, dim=1)

    return accuracy_score(y,y_hat.numpy())

num_layers=2

model=ThreeLayerLSTM(input_size=64, hidden_size=5, output_size=3,num_layers=num_layers).to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Sparse categorical cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_games_tensor = np.memmap("../data/numpy_objects/train_games_tensor.dat", dtype='float64', mode='r',
                               shape=(240000, 64, 64))
train_games_y = np.load("../data/numpy_objects/train_target.npy")

val_games_tensor = np.memmap("../data/numpy_objects/val_games_tensor.dat", dtype='float64', mode='r',
                               shape=(30000, 64, 64))
val_games_y = np.load("../data/numpy_objects/val_target.npy")

epochs = 25
batch_size = 1000
batch_num = train_games_tensor.shape[0] // batch_size
training_report={
    "loss":[],"accuracy":[],"validation loss":[],"validation accuracy":[]

                 }


for epoch in range(epochs):
    model.to(device)
    losses=[]
    acc=[]
    for i in range(batch_num):
        model.train()
        model.zero_grad()

        input = train_games_tensor[i * batch_size:(i + 1) * batch_size]
        input=np.array(input)
        input = torch.from_numpy(input).to(torch.float32).to(device)


        target = train_games_y[i * batch_size:(i + 1) * batch_size]
        target = torch.from_numpy(target).to(torch.long).to(device)

        output = model(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_acc=accuracy_score_check(target,output)

        losses.append(loss.item())
        acc.append(train_acc)

    model.eval()
#    model.cpu()

    with torch.no_grad():

        val_input = val_games_tensor[:]
        input = torch.from_numpy(val_input).to(torch.float32).to(device)

        target = val_games_y[:]
        target = torch.from_numpy(target).to(torch.long).to(device)

        output = model(input)

        val_loss = criterion(output, target)
 #       print("error")
        acc = accuracy_score_check(target, output)


    training_report["loss"].append(np.mean(losses))
    training_report["accuracy"].append(np.mean(acc))
    training_report["validation loss"].append(val_loss.item())
    training_report["validation accuracy"].append(acc)

    print(f"Epoch {epoch + 1}/{epochs}.. the loss is: {np.mean(losses)},the val loss is: {val_loss.item()}, the accuracy is {np.mean(acc)} and the val accuracy is: {acc} ")


with open(f"../reports/{num_layers}_layers_lstm_training_report.json", "w") as outfile:
    json.dump(training_report, outfile)

torch.save(model.state_dict(), f'../models/{num_layers}_layers_lstm_model_weights.pth')