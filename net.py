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
        self.n_hidden_layers = n_hidden_layers

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
 

class BaggingEnsemble():
    def __init__(self, models, accuracies):
        #assert len(models) == len(pred_accuracies)
        self.models = models
        self.accuracies = accuracies

    def __call__(self, x):
        prediction = 0
        for i, e in enumerate(self.models):
             prediction += e(x) * (self.accuracies[i] / sum(self.accuracies)) 
        
        return prediction

class StackingEnsemble(torch.nn.Module):
    """
        A stacking ensemble is a nn that trains on the outputs of weaker models. 
    """
    def __init__(self, models, input_size,  n_hidden_layers, hidden_size, activation_function):
        if len(models) == 0:
            raise Exception("Invalid number of models for stacking ensemble (0)")
        
        super(StackingEnsemble, self).__init__()
        layers = []
        self.models = models
        layers.append(torch.nn.Linear(input_size, hidden_size))
        
        for _ in range(n_hidden_layers):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation_function())
        layers.append(torch.nn.Linear(hidden_size, 1))
        #layers.append(torch.nn.Linear())
        self.n = torch.nn.Sequential(*layers)


    def forward(self, x): 
        return self.n(x)
    

    def construct_input(self, x):
        weak_models_predictions = []
        ret = []

        inp = torch.autograd.Variable(torch.tensor(x).type(torch.FloatTensor), requires_grad=True)
        for e in self.models:
            weak_models_predictions.append(e(inp).detach().numpy())
        for i in range(len(weak_models_predictions[0])):
            preds = []
            for j in range(len(self.models)):
                preds.append(weak_models_predictions[j][i].item(0))
            ret.append(preds)
        
        return ret 

    def construct_prediction_input(self, x):
        weak_models_predictions = []

        inp = torch.autograd.Variable(torch.tensor(x).type(torch.FloatTensor), requires_grad=True)
        for e in self.models:
            weak_models_predictions.append(e(inp).detach().numpy())
        
        tr = torch.transpose(torch.tensor(weak_models_predictions), 0, 1)
        return tr
    
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, n_hidden_layers, hidden_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, n_hidden_layers, nonlinearity='relu')
        self.hidden_state = torch.autograd.Variable(torch.zeros(n_hidden_layers, input_size, hidden_size))
        self.cell_state = torch.autograd.Variable(torch.zeros(n_hidden_layers, input_size, hidden_size))
        self.output_layer = torch.nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.input_size = input_size

    def forward(self, x):
        hidden_state = torch.zeros(self.n_hidden_layers, self.input_size, self.hidden_size)
        cell_state = hidden_state
        out, _ = self.rnn(x, hidden_state)
        #out = out.contiguous().view(-1, self.hidden_size)
        out = self.output_layer(x)
        
        return out


#class LSTM(nn.Module):
   # def __init__(self, input_size, n_hidden_layers, hidden_size, )

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, input):
        return self.linear(input)

def train_rnn(model, x_train, y_train, x_test, y_test, learning_rate=1e-6, epochs=50000):
    losses = []
    train_loss = 0
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    x = torch.autograd.Variable(torch.tensor(x_train).type(torch.FloatTensor), requires_grad=True)
    y = torch.autograd.Variable(torch.tensor(y_train).type(torch.FloatTensor), requires_grad=True)
    x = x.unsqueeze(0)
    for t in range(epochs):
        print(x.size())
        #print(x.unsqueeze(0).shape)   
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
            output = model(torch.tensor(x_test[i]).type(torch.FloatTensor))
            pred_values.append(output.item())
    return (losses, pred_values)

def train_stacking(model, x_train, y_train, x_test, y_test, learning_rate=1e-6, epochs=10000):
    
    if len(model.models) == 0:
        print("No models provided!")
        return
    losses = []
    train_loss = 0
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    x = torch.autograd.Variable(torch.tensor(x_train).type(torch.FloatTensor), requires_grad=True)
    y = torch.autograd.Variable(torch.tensor(y_train).type(torch.FloatTensor), requires_grad=True)
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
    
    pred_values = []
    with torch.no_grad():
        model.eval()
        for i in range(len(x_test)):
            output = model(torch.tensor(x_test[i]).type(torch.FloatTensor))
            pred_values.append(output.item())
    
    return (losses, pred_values)

"""
    Performs training and testing
    Returns a tuple of format (train_losses, pred_values)
"""
def train(model, x_train, y_train, learning_rate=1e-6, epochs=50000):
    
    losses = []
    train_loss = 0
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    x = torch.autograd.Variable(torch.tensor(x_train).type(torch.FloatTensor), requires_grad=True)
    y = torch.autograd.Variable(torch.tensor(y_train).type(torch.FloatTensor), requires_grad=True)
    for t in range(epochs):
        pred = model(x)
        loss = loss_function(pred, y)
        opt.zero_grad()    
        loss.backward()
        opt.step()
        if t % 5 == 0:
            losses.append(loss.item())
            #train_loss = loss.item()
            #print("Loss: ", loss)
    
    return losses


def eval(model, x_test, y_test):
    
    pred_values = []
    with torch.no_grad():
        model.eval()
        for i in range(len(x_test)):
            output = model(torch.tensor(x_test[i]).type(torch.FloatTensor))
            pred_values.append(output.item())

    return pred_values


def predict(model, inp):
    """
        Make a single prediction
    """
    with torch.no_grad():
       model.eval()
       return model(torch.tensor(inp).type(torch.FloatTensor))

