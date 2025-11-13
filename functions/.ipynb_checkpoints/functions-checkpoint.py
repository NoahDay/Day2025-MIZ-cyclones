import numpy as np
from scipy import signal

def cyclic_moving_av( a, n= 30, win_type= 'boxcar' ):
    # https://stackoverflow.com/questions/36074074/smooth-circular-data
  window= signal.get_window( win_type, n, fftbins=False ).reshape( (1,n) )
  shp_a= a.shape
  b= signal.convolve2d( a.reshape( ( np.prod( shp_a[:-1], dtype=int ), shp_a[-1] ) ), 
                        window, boundary='wrap', mode='same' )
  return ( b / np.sum( window ) ).reshape( shp_a )