import h5py
import numpy as np
import tables
import hdf5plugin
#HDF5的读取：
f = h5py.File('train.name.h5','r')   #打开h5文件
a = h5py.File('train.apiseq.h5','r') 
# 可以查看所有的主键

print(f['indices'],f['indices'][1],f['indices'][2])
print(type(f['indices']).__name__)
print(type(a['indices']).__name__)
print(a['indices'])
print(a['indices'].name)
print(a['indices'].dtype)
print(f['phrases'])
