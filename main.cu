#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "src/simulation.cu"

#define N_IN 50
#define N_REC 250
#define N_WINDOW 10000
#define TIMESTEPS 20000

#define THRESHOLD 0.3
#define VOLT_TAU 20.0
#define VOLT_COEFF exp(-1.0/VOLT_TAU)
#define REF_PERIOD 2

#define W_IN_SIZE N_IN*N_REC
#define W_REC_SIZE N_REC*N_REC
#define NET_STATE_SIZE N_WINDOW*N_REC

int main() {

    Simulation mySim = Simulation(N_REC, N_IN, REF_PERIOD, THRESHOLD, VOLT_TAU, N_WINDOW);
    mySim.allocate();
    mySim.copyToDevice();
    mySim.simulate(TIMESTEPS);
    mySim.copyFromDevice();
    mySim.free();

    //view a subset of spike trains and voltages
    for(size_t i = N_WINDOW - 100; i < N_WINDOW; ++i) {
        //printf("%2d", i);
        cout << setw(5) << i;
        for(size_t j = 0; j < 10; ++j) {
            //printf("  ");
            //printf("%4.2f", volts[N_REC*i + j]);
            cout << "  ";
            cout << fixed << setprecision(2) << setw(5) << mySim.voltages[N_REC*i + j];
        }
        //printf("    ");
        cout << "    ";
        for(size_t j = 0; j < 10; ++j) {
            //printf("  ");
            //printf("%4.2f", spikes[N_REC*i + j]);
            cout << "  ";
            cout << fixed << setprecision(2) << setw(5) << mySim.spikes[N_REC*i + j];
        }
        cout << endl;
    }

    return 0;
}
