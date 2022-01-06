# -*- coding: utf-8 -*-
#stacking + splitting

def StackedData(data):
    """
    

    Parameters
    ----------
    data : dict type object, with keys 'features' and 'labels', containing the images and the corresponding 
    masks as lists of numpy arrays
    
    Raises
    ------
    ValueError if the argument doesn't all of the characteristics listed above

    Returns
    -------
    stacked : dict type object containing a numpy array which groups the data passed to the function

    """
    if type(data) is dict:
        stacked = {'features': [], 'labels':[]}
        for i in range(len(data['features'])):
            for j in range(len(data['features'][i][0, 0, :])):
                stacked['features'].append(data['features'][i][:,:,j])
                
        for i in range(len(data['labels'])):
            for j in range(len(data['labels'][i][0, 0, :])):
                stacked['labels'].append(data['labels'][i][:,:,j])
        
        return stacked
    else:
        raise ValueError("A dict type object with 2 keys was expected, a/an {} was given instead".format(type(data)))

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
        train_data['features'].append(stacked_data['features'][i])
        train_data['labels'].append(stacked_data['labels'][i])
    for i in range(len(val_split)):
        val_data['features'].append(stacked_data['features'][(train_split + i)])
        val_data['labels'].append(stacked_data['labels'][(train_split + i)])
    for i in range(len(test_split)):
        test_data['features'].append(stacked_data['features'][(train_split + val_split + i)])
        test_data['labels'].append(stacked_data['labels'][(train_split + val_split + i)])
    
    return train_data, val_data, test_data
