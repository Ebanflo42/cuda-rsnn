#include <stdio.h>
//#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "src/simulation.cu"

#define N_IN 4
#define N_REC 8
#define N_WINDOW 20
#define TIMESTEPS 20

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

    for(size_t i = 0; i < N_WINDOW; ++i) {
        printf("%2d", i);
        //std::cout << std::setw(4) << i
        for(size_t j = 0; j < N_REC; ++j) {
            printf("  ");
            printf("%4.2f", mySim.voltages[N_REC*i + j]);
            //std::cout << "  "
            //std::cout << setw(4) << volts[N_REC*i + j];
        }
        printf("    ");
        for(size_t j = 0; j < N_REC; ++j) {
            printf("  ");
            printf("%4.2f", mySim.spikes[N_REC*i + j]);
            //std::cout << "  "
            //std::cout << setw(4) < spikes[N_REC*i + j];
        }
        printf("\n");
    }

    return 0;
}
