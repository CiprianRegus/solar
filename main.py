import faulthandler; faulthandler.enable()

import torch
import torch.nn
import torch.autograd
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
import threading
import sys
import os
import numpy as np
import math as math
import extract_util as eu
import net as network
import data_util as du
import function as lossf
import time_util as time
from importlib import reload
import copy

INPUT_SIZE = 4
OUTPUT_SIZE = 1
EPOCHS = 200
LEARNING_RATE = 1e-2
SECONDS_IN_DAY = 60*60*24
INTERVAL_OFFSET = 1
DAY_OFFSET = 4*24
NEXT_DAY_OFFSET = 24*4

libs = ['matplotlib', 'torch', 'pandas']
def check_installed_libs(libs):
    for lib in libs:
        if lib not in sys.modules:
            
            if (spec := importlib.util.find_spec(lib)) is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules[lib] = module
                spec.loader.exec_module(module)
                print(f"{lib!r} has been imported")
            else:
                raise BaseException(lib)
                
    
try: 
    check_installed_libs(libs)
except BaseException as e: 
    print(f"Package {e.args[0]!r} not found")
    os._exit

reload(network)
reload(lossf)
reload(time)
reload(du)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
print(torch.cuda.is_available())


def split_inverters():
    plant1Data = pd.read_csv("plant1Data") # Reimport the processed data
    """
        There is a difference between the production reported by each inverter, so the prediction will be made for each of them
    """
    inverters = set()
    for e in plant1Data['SOURCE_KEY']:
        inverters.add(e)
    """
        Split data in subsets for each inverter
    """
    inverter_subsets = []
    # The set is sorted, so each time we'll have the subsets in the same order
    for e in sorted(inverters):
        subset = plant1Data[plant1Data['SOURCE_KEY'] == e]
        inverter_subsets.append(subset)
        num_records = len(subset["AMBIENT_TEMPERATURE"])
        subset.index = [x for x in range(num_records)]
    return inverter_subsets
"""
    data = (data - mean(data)) / std(data)
"""
def normalize_data(inverter_subsets):

    x_train, y_train, x_test, y_test = du.split_train_test(inverter_subsets[11], 0.8 
                                    , ["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC", "PREVIOUS_DAY_AC", "DC_POWER", "AC_POWER"], 
                                    ["DC_POWER", "AC_POWER", "DAILY_YIELD"])
    #Data is normalized
    test_offset = x_test.first_valid_index()
    for col in ["DC_POWER", "AC_POWER"]:
        test_std = normalize_array(x_train[col].values, method="mean_std")
        for i in range(len(test_std)):
            y_train.loc[i, col] = float(test_std[i].item())
            x_train.loc[i, col] = float(test_std[i].item())

        test_std = normalize_array(y_test[col].values, method="mean_std")
        for i in range(test_offset, len(test_std) + test_offset):
            y_test.loc[i, col] = float(test_std[i - test_offset].item())
            x_test.loc[i, col] = float(test_std[i - test_offset].item())

    for col in ["AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC", "PREVIOUS_DAY_AC"]:
        test_std = normalize_array(x_train[col].values, method="mean_std")
        for i in range(len(test_std)):
            x_train.loc[i, col] = float(test_std[i].item())

        test_std = normalize_array(x_test[col].values, method="mean_std")
        for i in range(test_offset, len(test_std) + test_offset):
            try:
                x_test.loc[i, col] = float(test_std[i - test_offset].item())
            except:
                print("INVALID INDEX: ", i)
    # transform seconds -> sin((seconds/seconds_in_day) * pi))
    for i in range(len(x_train["SECONDS"])):
        x_train.loc[i, "SECONDS"] = math.sin((x_train.loc[i, "SECONDS"] / SECONDS_IN_DAY) * math.pi )
    for i in range(test_offset, len(x_test["SECONDS"]) + test_offset):
        try:
            x_test.loc[i, "SECONDS"] = math.sin((x_test.loc[i, "SECONDS"] / SECONDS_IN_DAY) * math.pi )
        except:
            print("INVALID INDEX: ", i)

    x_train = x_train[["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_AC"]].values
    y_train = y_train[["AC_POWER"]].values
    x_test = x_test[["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_AC"]].values
    y_test = y_test[["AC_POWER"]].values

    return x_train, y_train, x_test, y_test


def normalize_array(arr, method="min_max"):
    """
        Returns a tensor of size (len(arr))
    """
    
    ret = torch.tensor(arr)
    if method == "min_max":
        ret -= torch.min(ret)
        ret /= torch.max(ret)
    elif method == "mean_std":
        ret -= torch.mean(ret)
        ret /= torch.std(ret)
    else:
        raise Exception("Invalid normalization method")

    return 1 + ret


def run_linear_regression(x_train, y_train, x_test, y_test):
    
    losses = []
    corr_values = [x.item() for x in y_test]
    linear_model = 0
    linear_model = network.LinearRegression(input_size=x_train.shape[1])
    if "--load_linear" in sys.argv:
        linear_model.load_state_dict(torch.load("models/linear_model.pt"))
        print("Loading")
    else:
        losses = network.train(linear_model, x_train, y_train, learning_rate=1e-1, epochs=10000) 
        torch.save(linear_model.state_dict(), "models/linear_model.pt")        
    pred_values = network.eval(linear_model, x_test, y_test)
    print("MAPE regresie: ", lossf.mape(pred_values, corr_values))

def run_mlp(x_train, y_train, x_test, y_test):
   
    """
        MLP
    """
    losses = []
    all_pred_values = []
    models = []
    accuracies = [] 
    stacking_input = []
    bagging_pred = []
    
    """
    rnn_model = network.RNNModel(input_size=x_train.shape[1], n_hidden_layers=1, hidden_size=x_train.shape[1])
    losses, pred_values = network.train_rnn(rnn_model, x_train, y_train, x_test, y_test, learning_rate=LEARNING_RATE, epochs=EPOCHS) 
    print("RNN final loss: ", losses[-1])
    print("RNN MAPE: ", lossf.mape(pred_values, [e.item() for e in y_train]))
    """
    for i in range(1, 6):
        mod = network.MLP(input_size=x_train.shape[1], n_hidden_layers=i, hidden_size= x_train.shape[1] , activation_function=torch.nn.SiLU)
        #mod.cuda()
        losses = network.train(mod, x_train, y_train, learning_rate=LEARNING_RATE, epochs=EPOCHS) 
        print("{} layers loss: {}".format(i, losses[-1]))
        pred_values = network.eval(mod, x_test, y_test)
        all_pred_values.append(pred_values)
        models.append((mod, losses[-1]))
    losses = min(models, key=lambda x: x[1])
    innacurate_models = []
    for i in range(len(models)):
        current_mape = lossf.mape(all_pred_values[i], [e.item() for e in y_test])
        """
        if current_mape > 15:
            innacurate_models.append(models[i])
        """
        accuracies.append(100 - current_mape)
        print("{} layers MAPE: {} ".format(i+1, current_mape))
        
    for e in innacurate_models:
        models.remove(e)
    models = [e[0] for e in models]
    print("{} models selected as input for stacking ensemble".format(len(models)))
    """
        Bagging ensemble
    
    models = [e[0] for e in models]
    
    bagging_model = network.BaggingEnsemble(models, accuracies)
    for i in range(len(x_test)):
        output = bagging_model(torch.tensor(x_test[i]).type(torch.FloatTensor))
        bagging_pred.append(output)
    print("Bagging ensemble MAPE: {.2%}".format(lossf.mape(bagging_pred, [e.item() for e in y_test])))
    """

    """
        Stacking ensemble
    """
    for i in range(len(all_pred_values[0])):
        inp = []
        for j in range(len(all_pred_values)):
            inp.append(all_pred_values[j][i])
        stacking_input.append(inp)
    stacking_input_train = stacking_input[:int(0.8 * len(stacking_input))]
    stacking_input_test = stacking_input[int(0.8*len(stacking_input)):]
    stacking_model = network.StackingEnsemble(models, input_size=len(stacking_input[0]), n_hidden_layers=3, hidden_size=len(stacking_input[0]), activation_function=torch.nn.SiLU)
    stacking_train = stacking_model.construct_input(x_train)
    stacking_test = stacking_model.construct_input(x_test)
    losses, pred_values = network.train_stacking(stacking_model, stacking_train, y_train, stacking_test, y_test, learning_rate=1e-2, epochs=25000) 
    print("Stacking ensemble MAPE: {} ".format(lossf.mape(pred_values, [e.item() for e in y_test])))

if __name__ == "__main__":

    inverter_subsets = split_inverters()
    inverter_subsets_copy = copy.deepcopy(inverter_subsets)
    x_train, y_train, x_test, y_test = normalize_data(inverter_subsets)
    thread_linear = threading.Thread(run_linear_regression(x_train, y_train, x_test, y_test))
    thread_mlp = threading.Thread(run_mlp(x_train, y_train, x_test, y_test))
    thread_linear.start()
    thread_mlp.start()
    thread_mlp.join()
    thread_linear.join()
