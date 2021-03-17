import torch
import math

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MLP(torch.nn.Module):
    def __init__(self, input_size, n_hidden_layers, hidden_size, activation_function):
        super(MLP, self).__init__()
        #self.weights = torch.nn.Parameter(torch.randn())
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        for _ in range(n_hidden_layers):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation_function())
        layers.append(torch.nn.Linear(hidden_size, 1))
        #layers.append(torch.nn.Linear())
        self.n = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.n(x)

class TestMLP(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(TestMLP, self).__init__()
        self.hidden1 = torch.nn.Linear(n_features, n_hidden)
        self.hidden2 = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden1(x))
        x = torch.nn.functional.relu(self.hidden2(x))
        x = self.predict(x)
        return x
    
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input):
        return self.linear(input)


"""
    Performs trainig and testing
    Returns a tuple of format (train_losses, pred_values)
"""
def train(model, x_train, y_train, x_test, y_test, learning_rate=1e-6, epochs=50000):
    losses = []
    train_loss = 0
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    x = torch.autograd.Variable(torch.tensor(x_train, device=dev).type(torch.cuda.FloatTensor), requires_grad=True)
    y = torch.autograd.Variable(torch.tensor(y_train, device=dev).type(torch.cuda.FloatTensor), requires_grad=True)
    for t in range(epochs):
        pred = model(x)
        loss = loss_function(pred, y)
        opt.zero_grad()    
        loss.backward()
        opt.step()
        if t == epochs - 1:
            print("Final loss: ", loss.item())
        if t % 5 == 0:
            losses.append(loss.item())
            #train_loss = loss.item()
            #print("Loss: ", loss)

    pred_values = []
    with torch.no_grad():
        model.eval()
        for i in range(len(x_test)):
            output = model(torch.tensor(x_test[i], device=dev).type(torch.FloatTensor))
            pred_values.append(output.item())

    return (losses, pred_values)