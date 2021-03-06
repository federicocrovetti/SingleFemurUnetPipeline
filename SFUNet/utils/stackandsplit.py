import numpy as np
import SimpleITK as sitk

def NormDict(data):
    """
    
    Parameters
    ----------
    data : dict type object, with keys 'features' and 'labels', containing the images and the corresponding 
    masks as lists of numpy arrays
    Returns
    -------
    norm : dict type object of the same kind of the input 'data', where each array has values in the [0,1] range
    """
    if type(data) is dict:
        if type(data['features'][0]) == np.ndarray:
            norm = {'features': [], 'labels':[]}
            for i in range(int(len(data['features']))):
                shift = (data['features'][i] + abs(np.min(data['features'][i])))
                norm_data = np.divide(shift, np.max(shift))
                norm['features'].append(norm_data)
                norm['labels'].append(data['labels'][i])
            return norm 
        else:
            raise ValueError("A dict type object with 2 keys, containing a series of arrays was expected, instead a/an {} was given".format(type(data['features'][0])))
    else:
        raise ValueError("A dict type object with 2 keys was expected, a/an {} was given instead".format(type(data)))

def StackedData(data):
    """
    
    Parameters
    ----------
    data : dict type object, with keys 'features' and 'labels', containing the images and the corresponding 
    masks as lists of numpy arrays, or SimpleItk.Image objects
    
    Raises
    ------
    ValueError if the argument doesn't all of the characteristics listed above
    Returns
    -------
    stacked : dict type object containing a numpy array which groups the data passed to the function
    """
    if type(data) is dict:
        if type(data['features'][0]) == sitk.Image:
            stacked = {'features': [], 'labels':[]}
            for i in range(len(data['features'])):
                for j in range(data['features'][i].GetSize()[2]):
                    arr_slice = sitk.GetArrayFromImage(data['features'][i][:,:,j])
                    stacked['features'].append(arr_slice)
                    lab_slice = sitk.GetArrayFromImage(data['labels'][i][:,:,j])
                    stacked['labels'].append(lab_slice)
        elif type(data['features'][0]) == np.ndarray:
            stacked = {'features': [], 'labels':[]}
            for i in range(len(data['features'])):
                for j in range(data['features'][i].shape[2]):
                    arr_slice = data['features'][i][:,:,j]
                    stacked['features'].append(arr_slice)
                    lab_slice = data['labels'][i][:,:,j]
                    stacked['labels'].append(lab_slice)
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
    stacked_data = NormDict(stacked_data)

    train_data = {'features': [], 'labels':[]}
    val_data = {'features': [], 'labels':[]}
    test_data = {'features': [], 'labels':[]}

    tot_samples = len(stacked_data['features'])
    train_split = round(tot_samples*train_percent)
    val_split = round(tot_samples*val_percent)
    test_split = round(tot_samples*test_percent)

    for i in range(train_split):
        train_data['features'].append(stacked_data['features'][i])
        train_data['labels'].append(stacked_data['labels'][i])
    for i in range(val_split):
        val_data['features'].append(stacked_data['features'][(train_split + i)])
        val_data['labels'].append(stacked_data['labels'][(train_split + i)])
    for i in range(test_split):
        test_data['features'].append(stacked_data['features'][(train_split + val_split + i)])
        test_data['labels'].append(stacked_data['labels'][(train_split + val_split + i)])

    return train_data, val_data, test_data
