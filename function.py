def mape(pred_values, corr_values):
    EPS = 0.01 # This is the min value that corr_values[i] can take, so division by zero can be avoided
    if len(pred_values) == 0:
        print("pred_values length is 0: ")
        return 1
    mape_sum = 0.0
    for i in range(len(pred_values)):
        mape_sum += abs((pred_values[i] - corr_values[i]) / (max(corr_values[i], EPS)))
    mape_sum /= len(pred_values) 
    return mape_sum * 100
