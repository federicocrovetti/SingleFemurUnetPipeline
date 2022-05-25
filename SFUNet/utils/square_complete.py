
def SquareComplete(dataset, req_size):
    """
    
    Parameters
    ----------
    dataset : dict type object with 'features' and 'labels' as keys containins sitk.Image objects
    req_size : (int, int) or (int, int, int) tuple with the minimum desired size the image is required 
    to be after the padding along x, y and, optionally z, axis.
    Returns
    -------
    pad_sizes : list of lists with the necessary padding sizes for the images to have the required sizes
    """
    if len(req_size) == 2:
        pad_sizes = []
        for i in range(len(dataset['features'])):
            size = dataset['features'][i].GetSize()
            pad_size = (req_size[0]-size[0], req_size[1]-size[1], 0)
            pad_size = list(pad_size)
            pad_sizes.append(pad_size)
        return pad_sizes
    
    elif len(req_size) == 3:
        pad_sizes = []
        for i in range(len(dataset['features'])):
            size = dataset['features'][i].GetSize()
            pad_size = (req_size[0]-size[0], req_size[1]-size[1], req_size[2]-size[2])
            pad_size = list(pad_size)
            pad_sizes.append(pad_size)
        return pad_sizes
