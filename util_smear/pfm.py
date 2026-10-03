import re
import numpy as np
import sys

def readPFM(file):
    """ Read PFM file
    Args:
        .pfm file
    Returns:
        data, scale
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True    
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    """ Write a IMAGE into pfm format

    Args:
        file (str): Output filename
        image : HxWx2 or HxWx3
        scale : Defaults to 1.

    """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        # raise Exception('Image dtype must be float32.')
        image = image.astype(np.float32)
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    # PFM header
    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    
    # Write width and height information
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)
    
def readSmearPFM(file):
    """ Read Smear PFM file 
        The data shape (H x W x 5)
        startx, starty, endx, endy, distance
    Args:
        .pfm file
    Returns:
        data, scale
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF5':
        pass   
    else:
        raise Exception('Not a PFM file for Smear annotation (H x W x 5 required).')

    dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    ### Smear annotation have shape (H x W x 5)
    shape = (height, width, 5) 

    # print(file, flush=True)
    # print(data.shape, flush=True)
    
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
    
def writeSmearPFM(file, image, scale=1):
    """ Write a H x W x 5 into pfm format

    Args:
        file (str): Output pfm filename
        image : data with shape (HxWx5)
        scale : Defaults to 1.

    """
    file = open(file, 'wb')
    

    if image.dtype.name != 'float32':
        # raise Exception('Image dtype must be float32.')
        image = image.astype(np.float32)
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 5: # [startX, startY, endX, endY, valid_mask]
        pass
    else:
        raise Exception('Image must have H x W x 5 dimensions.')

    # PFM header
    file.write('PF5\n'.encode()) # Custom header for 5-channel data
    
    # Write width and height information
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)