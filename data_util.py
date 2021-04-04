def split_train_test(dataset, split_ratio, train_columns, validation_columns):
    for e in train_columns + validation_columns:
        if e not in dataset.columns:
            raise Exception("Column not found in dataset: ",  e)
   
    if split_ratio > 1 or split_ratio < 0:
        raise ValueError("Invalid split ratio value: ", split_ratio)
    training_data = dataset[:int(split_ratio * len(dataset))]
    testing_data = dataset[int(split_ratio * len(dataset)):]
    x_train = training_data[train_columns]
    y_train = training_data[validation_columns]
    x_test = testing_data[train_columns]
    y_test = testing_data[validation_columns]

    return (x_train, y_train, x_test, y_test)

