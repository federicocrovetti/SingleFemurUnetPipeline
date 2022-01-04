# -*- coding: utf-8 -*-
#stacking + splitting

def StackedData(data):
    """
    

    Parameters
    ----------
    data : dict type object containing the images and the corresponding masks in a numpy array form

    Returns
    -------
    stacked : dict type object containing a numpy array which groups the data passed to the function

    """
    
    if type(data) is dict:
        stacked = {'features': [], 'labels':[]}
        values = list(data.values())
        for i in range(len(values[0])):
            stacked['features'].append(values[0][i])
            stacked['labels'].append(values[1][i])
        
        return stacked
    else:
        raise ValueError("A dict type object with 2 keys was expected, a {} was given instead".format(type(data)))


def Split(data, train_percent, val_percent, test_percent):
    """
    

    Parameters
    ----------
    data : dict type object, with keys 'features' and 'labels', corresponding to a list of arrays.
    train_percent : int or float. The percentage of data that is intended for the training process.
    val_percent : int or float. The percentage of data that is intended for the validation process.
    test_percent : int or float. The percentage of data that is intended for the testing process.

    Returns
    -------
    A partition of the stacked 'data', divided between train, validation and test sets.
    This function depends on 'StackedData' function.

    """
    if (train_percent + val_percent + test_percent) != 1:
        raise ValueError('The dataset splitting does not add up to 1')
    stacked_data = StackedData(data)
        
    train_data = {'features': [], 'labels':[]}
    val_data = {'features': [], 'labels':[]}
    test_data = {'features': [], 'labels':[]}
    
    tot_samples = len(stacked_data['features'])
    train_split = tot_samples*train_percent
    val_split = tot_samples*val_percent
    test_split = tot_samples*test_percent
    
    for i in range(len(train_split)):
        train_data['features'][i] = stacked_data['features'][i]
        train_data['labels'][i] = stacked_data['labels'][i]
    for i in range(len(val_split)):
        val_data['features'][train_split + i] = stacked_data['features'][train_split + i]
        val_data['labels'][train_split + i] = val_data['labels'][train_split + i]
    for i in range(len(test_split)):
        test_data['features'][train_split + val_split + i] = stacked_data['features'][train_split + val_split + i]
        test_data['labels'][train_split + val_split + i] = test_data['labels'][train_split + val_split + i]
    
    return train_data, val_data, test_data
