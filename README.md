# cuda-rsnn
Simulating recurrent spiking neural networks with cuda.

## Testing the python interface

Clone this repository onto a cuda-capable machine:

```
git clone https://github.com/Ebanflo42/cuda-rsnn
```

Compile the shared object file:

```
nvcc src/simulation.cu -o src/simulation.so -shared -Xcompiler -fPIC -std=c++11
```

Running this script will simulate a LIF network of 250 neurons for 10000 timesteps with random (uniform) weights and input currents, and then print a subset of the voltages and spikes:

```
python pytest.py
```

Note that the voltages are in a very unrealistic range, since no initialization or regularization tricks have been used.
