#!/usr/bin/python

#%%
from config.init import *

#%%
passengers = manifest_data_0['passenger_info']
hashify(passengers, namespace='passenger')

#%%
