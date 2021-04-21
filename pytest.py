from ctypes import *
import os
import numpy as np

N_IN = 5
N_REC = 250
N_WINDOW = 10000
TIMESTEPS = 10000

THRESHOLD = 0.3
VOLT_TAU = 20.0
REF_PERIOD = 2

if __name__ == '__main__':

    simlib = cdll.LoadLibrary(os.getcwd() + "/src/simulation.so")

    simlib.new_simulation.argtypes = [c_int, c_int, c_int, c_float, c_float, c_int, c_void_p, c_void_p]
    simlib.new_simulation.restype = c_void_p
    simlib.allocate_simulation.argtypes = [c_void_p]
    simlib.copy_to_device.argtypes = [c_void_p]
    simlib.run_simulation.argtypes = [c_void_p, c_int]
    simlib.copy_from_device.argtypes = [c_void_p]
    simlib.free_simulation.argtypes = [c_void_p]

    np_volts = np.zeros((N_WINDOW*N_REC), dtype=np.float32)
    np_spikes = np.zeros((N_WINDOW*N_REC), dtype=np.float32)

    my_sim = simlib.new_simulation(N_REC, N_IN, REF_PERIOD, THRESHOLD, VOLT_TAU, N_WINDOW, np_volts.ctypes.data, np_spikes.ctypes.data);
    simlib.allocate_simulation(my_sim)
    simlib.copy_to_device(my_sim)
    simlib.run_simulation(my_sim, TIMESTEPS)
    simlib.copy_from_device(my_sim);
    simlib.free_simulation(my_sim);

    np_volts = np.reshape(np_volts, (N_WINDOW, N_REC))
    np_spikes = np.reshape(np_spikes, (N_WINDOW, N_REC))

    print(np_volts[-100:, -8:])
    print(np_spikes[-100:, -8:])
