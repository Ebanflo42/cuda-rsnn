from ctypes import *
import os
import numpy as np

N_IN = c_int(4)
N_REC = c_int(8)
N_WINDOW = c_int(20)
TIMESTEPS = c_int(20)

THRESHOLD = c_float(0.3)
VOLT_TAU = c_float(20.0)
REF_PERIOD = c_int(2)

if __name__ == '__main__':

    mainlib = cdll.LoadLibrary("/home/eben/Projects/cuda-rsnn/main.so")
    mainlib.main()

    simlib = cdll.LoadLibrary("/home/eben/Projects/cuda-rsnn/src/simulation.so")

    my_sim = simlib.new_simulation(N_REC, N_IN, REF_PERIOD, THRESHOLD, VOLT_TAU, N_WINDOW);
    print(' got here ')
    simlib.allocate_simulation(my_sim)
    print(' got here ')
    simlib.copy_to_device(my_sim)
    print(' got here ')
    simlib.run_simulation(my_sim, TIMESTEPS)
    print(' got here ')
    simlib.copy_from_device(my_sim);
    print(' got here ')
    simlib.free_simulation(my_sim);

    print(my_sim.voltages)
