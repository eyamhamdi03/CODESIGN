

import pyopencl as cl

for platform in cl.get_platforms():
    for device in platform.get_devices():
        print ("===============================================================")
        print ("Platform name:", platform.name)
        print ("Platform vendor:", platform.vendor)
        print ("---------------------------------------------------------------")
        print ("Device name:", device.name)
        print ("Device type:", cl.device_type.to_string(device.type))
        print ("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print ("Device compute units:", device.max_compute_units)