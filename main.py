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
EPOCHS = 40000
LEARNING_RATE = 1e-3
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
    for e in inverters:
        subset = plant1Data[plant1Data['SOURCE_KEY'] == e]
        inverter_subsets.append(subset)
        num_records = len(subset["AMBIENT_TEMPERATURE"])
        subset.index = [x for x in range(num_records)]
    
    return inverter_subsets
"""
    data = (data - mean(data)) / std(data)
"""
def normalize_data(inverter_subsets):

    x_train, y_train, x_test, y_test = du.split_train_test(inverter_subsets[0], 0.8 
                                    , ["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC", "PREVIOUS_DAY_AC", "DC_POWER", "AC_POWER"],                                          
                                    ["DC_POWER", "AC_POWER", "DAILY_YIELD"])

    #Test data is normalized
    test_offset = x_test.first_valid_index()
    for col in ["DC_POWER", "AC_POWER"]:
        test_std = torch.tensor(x_train[col].values)
        test_std -= torch.mean(test_std)
        test_std = test_std/torch.std(test_std)

        for i in range(len(test_std)):
            y_train.loc[i, col] = float(test_std[i].item())
            x_train.loc[i, col] = float(test_std[i].item())

        test_std = torch.tensor(y_test[col].values)
        test_std -= torch.mean(test_std)
        test_std = test_std/torch.std(test_std)
        for i in range(test_offset, len(test_std) + test_offset):
            y_test.loc[i, col] = float(test_std[i - test_offset].item())
            x_test.loc[i, col] = float(test_std[i - test_offset].item())

    for col in ["AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC", "PREVIOUS_DAY_AC"]:
        test_std = torch.tensor(x_train[col].values)
        test_std -= torch.mean(test_std)
        test_std = test_std/torch.std(test_std)
        for i in range(len(test_std)):
            x_train.loc[i, col] = float(test_std[i].item())

        test_std = torch.tensor(x_test[col].values)
        test_std -= torch.mean(test_std)
        test_std = test_std/torch.std(test_std)
        for i in range(test_offset, len(test_std) + test_offset):
            try:
                x_test.loc[i, col] = float(test_std[i - test_offset].item())
            except:
                print("INVALID INDEX: ", i)
    # transform seconds -> sin((seconds/seconds_in_day) * pi))
    for i in range(len(x_train["SECONDS"])):
        x_train.loc[i, "SECONDS"] = math.sin((x_train.loc[i, "SECONDS"] / SECONDS_IN_DAY) * math.pi / 2)
    for i in range(test_offset, len(x_test["SECONDS"]) + test_offset):
        try:
            x_test.loc[i, "SECONDS"] = math.sin((x_test.loc[i, "SECONDS"] / SECONDS_IN_DAY) * math.pi / 2)
        except:
            print("INVALID INDEX: ", i)

    x_train = x_train[["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC"]].values
    y_train = y_train[["DC_POWER"]].values
    x_test = x_test[["SECONDS", "AMBIENT_TEMPERATURE", "IRRADIATION", "PREVIOUS_DAY_DC"]].values
    y_test = y_test[["DC_POWER"]].values

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

    return ret


def run_linear_regression(inverter_subsets):

    x_train, y_train, x_test, y_test = normalize_data(inverter_subsets)

    linear_model = network.LinearRegression(INPUT_SIZE, OUTPUT_SIZE)
    #linear_model.cuda()
    crit = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(linear_model.parameters(), lr=LEARNING_RATE)
    # Testarea regresiei multiple cu datele de la primul invertor
    losses = []
    
    for epoch in range(EPOCHS):
        temp_input = torch.autograd.Variable(torch.tensor(x_train).type(torch.FloatTensor), requires_grad=True)
        target = torch.autograd.Variable(torch.tensor(y_train).type(torch.FloatTensor), requires_grad=True)
        optimizer.zero_grad()
        outputs = linear_model(temp_input)
        loss = crit(outputs, target)
        loss.backward()
        optimizer.step()
        if epoch == EPOCHS-1:
            print("Final loss: ", loss.item())
        if(epoch+1) % 100 ==0: 
            print('epoch [%d/%d], Loss: %.4f' % (epoch +1, EPOCHS, loss.item()))
            losses.append(loss.item())
            #print('epoch [%d/%d], Loss: %.4f' % (epoch +1, EPOCHS, losses[-1]))
	
    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(losses))], losses)
    ax.set(xlabel='epoch', ylabel='loss', title='Loss')
    ax.grid()
    
    pred_values = []
    corr_values = []
    with torch.no_grad():
        linear_model.eval()
        for i in range(len(x_test)):
            output = linear_model(torch.tensor(x_test[i]).type(torch.FloatTensor))
            corr_values.append(y_test[i].item())
            pred_values.append(output.item())


    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(pred_values))], pred_values)
    ax.set(xlabel='epoch', ylabel='DC_POWER', title='Prediction')
    ax.grid()

    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(corr_values))], corr_values)
    ax.set(xlabel='epoch', ylabel='loss', title='Correct Value')
    ax.grid()


    diff = []
    for i, e in enumerate(corr_values):
        diff.append(abs(e - pred_values[i]))
    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(diff))], diff)
    ax.set(xlabel='epoch', ylabel='Diff', title='Difference')
    ax.grid()

    print("MAPE: ", lossf.mape(pred_values, corr_values))

def run_mlp(inverter_subsets):
   
    print("Running mlp")
    print(inverter_subsets[0])
    x_train, y_train, x_test, y_test = normalize_data(inverter_subsets)
 
    """
        MLP
    """
    losses = []
    all_pred_values = []
    models = []
    accuracies = [] 
    stacking_input = []
    bagging_pred = []

    for i in range(1, 6):
        mod = network.MLP(input_size=x_train.shape[1], n_hidden_layers=i, hidden_size= x_train.shape[1], activation_function=torch.nn.SiLU)
        #mod.cuda()
        losses, pred_values = network.train_mlp(mod, x_train, y_train, x_test, y_test, learning_rate=LEARNING_RATE, epochs=EPOCHS) 
        all_pred_values.append(pred_values)
        models.append((mod, losses[-1]))
    losses = min(models, key=lambda x: x[1])

    """
    print(mod.parameters())
    pred_values = []
    corr_values = []
    with torch.no_grad():
        mod.eval()
        test_loss = 0
        for i in range(len(x_test)):
            output = mod(torch.tensor(x_test[i]).type(torch.FloatTensor))
            corr_values.append(y_test[i].item())
            pred_values.append(output.item())
    corr_values = [x.item() for x in y_test]
    diff = []
    for i, e in enumerate(corr_values):
        diff.append(abs(e - pred_values[i]))
    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(pred_values))], pred_values)
    ax.set(xlabel='epoch', ylabel='Predicted value', title='MLP-3 prediction')
    ax.grid()

    fig, ax = plt.subplots()
    ax.plot([x for x in range(0, len(diff))], diff)
    ax.set(xlabel='epoch', ylabel='Diff', title='MLP-3 Difference')
    ax.grid()
    """

    for i, e in enumerate(models):
        current_mape = lossf.mape(all_pred_values[i], [e.item() for e in y_test])
        if current_mape < 20:
            models.remove(e)
        accuracies.append(100 - current_mape)
        print("{0} layers: ".format(i+1), current_mape)

    """
        Bagging ensemble
    """
    
    models = [e[0] for e in models]

    bagging_model = network.BaggingEnsemble(models, accuracies)
    for i in range(len(x_test)):
        output = bagging_model(torch.tensor(x_test[i]).type(torch.FloatTensor))
        bagging_pred.append(output)
    print("Bagging ensemble MAPE: ", lossf.mape(bagging_pred, [e.item() for e in y_test]))
    
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
    losses, pred_values = network.train_stacking(mod, x_train, y_train, x_test, y_test, learning_rate=LEARNING_RATE, epochs=EPOCHS) 
    print("Stacking ensemble MAPE: ", lossf.mape(pred_values, [e.item() for e in y_test]))

def run_all():

    plant1Data = pd.read_csv("/home/xvr/Documents/licenta/Load forecasting/india/Plant_1_Generation_Data.csv")
    plant1Weather = pd.read_csv(r"/home/xvr/Documents/licenta/Load forecasting/india/Plant_1_Weather_Sensor_Data.csv")
    plant2Data = pd.read_csv(r"/home/xvr/Documents/licenta/Load forecasting/india/Plant_2_Generation_Data.csv")
    plant2Weather = pd.read_csv(r"/home/xvr/Documents/licenta/Load forecasting/india/Plant_2_Weather_Sensor_Data.csv")
    print(plant1Data)

    np.random.seed(19680801)
    plt.rcdefaults()

    daytime_dc_power = plant1Data[plant1Data.DATE_TIME == '15-05-2020 00:00']
    dc_power = plant1Data[['DATE_TIME', 'DC_POWER']]
    #plt.hist(dc_power, histtype='step')

    # Production by hour
    production = {}
    same_hour = 0
    for record in zip(dc_power['DATE_TIME'], dc_power['DC_POWER']):
        hour = time.get_hour(record[0])
        if hour == "00":
            same_hour += 1
        if hour not in production.keys():
            production[hour] = int(record[1])
        else:
            production[hour] += int(record[1])
    for key in production.keys():
        production[key] //= same_hour
    print(production)

    fig, ax = plt.subplots()
    ax.plot(production.keys(), production.values())
    ax.set(xlabel='ora', ylabel='energie produsa (W)', title='Productia in functie de ora')
    ax.grid()

    # Solar irradiation throughout the day

    weather1 = plant1Weather[['DATE_TIME', 'AMBIENT_TEMPERATURE']]
    print(weather1['AMBIENT_TEMPERATURE'])
    same_hour = 0
    temps1 = {}

    for record in zip(weather1['DATE_TIME'], weather1['AMBIENT_TEMPERATURE']):
        hour = time.get_hour(record[0])
        if hour == "00":
            same_hour += 1
        if hour not in temps1.keys():
            temps1[hour] = record[1]
        else:
            temps1[hour] += record[1]
    for key in temps1.keys():
        temps1[key] /= same_hour
        pass
    print(temps1)
    fig, ax = plt.subplots()
    ax.plot(temps1.keys(), temps1.values())
    ax.set(xlabel='ora', ylabel='Temperatura', title='Temperatura in functie de ora')
    ax.grid()

    # In case data was not processed (date_time split, and merging production and weather data) 
    """
    d = eu.columns_of_interrest(plant1Data, plant1Weather)
    d = eu.add_offset_columns(d, DAY_OFFSET)
    d.to_csv('plant1Data')
    """
    plant1Data = pd.read_csv("plant1Data") # Reimport the processed data
    print(plant1Data)


    """
        There is a difference between the production reported by each inverter, so the prediction will be made for each of them
    """
    inverters = set()
    for e in plant1Data['SOURCE_KEY']:
        inverters.add(e)
    print(inverters)

    """
        Split data in subsets for each inverter
    """

    inverter_subsets = []

    for e in inverters:
        subset = plant1Data[plant1Data['SOURCE_KEY'] == e]
        inverter_subsets.append(subset)
        num_records = len(subset["AMBIENT_TEMPERATURE"])
        subset.index = [x for x in range(num_records)]
    print(inverter_subsets)



    run_linear_regression(inverter_subsets)
if __name__ == "__main__":
    inverter_subsets = split_inverters()
    inverter_subsets_copy = copy.deepcopy(inverter_subsets)
    #run_linear_regression(inverter_subsets)
    #run_mlp(inverter_subsets_copy)
    thread_linear = threading.Thread(run_linear_regression(inverter_subsets))
    thread_mlp = threading.Thread(run_mlp(inverter_subsets_copy))
    thread_linear.start()
    thread_mlp.start()
