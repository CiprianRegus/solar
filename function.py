def mape(pred_values, corr_values):
    mape_sum = 0.0
    for i in range(len(pred_values)):
        mape_sum += abs((pred_values[i] - corr_values[i]) / corr_values[i])
    mape_sum /= len(pred_values)
    return mape_sum * 100
