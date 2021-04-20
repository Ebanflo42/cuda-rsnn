#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <vector>

using namespace std;

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

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

__global__ void network_step(float voltages[NET_STATE_SIZE],
                             float spike_trains[NET_STATE_SIZE],
                             int refractory_periods[NET_STATE_SIZE],
                             float in_currents[N_WINDOW*N_IN],
                             float weight_in[W_IN_SIZE],
                             float weight_rec[W_REC_SIZE],
                             int t) {

    int pre = blockIdx.x*blockDim.x + threadIdx.x;
    int post = blockIdx.y*blockDim.y + threadIdx.y;

    int last_t = (t - 1)%N_WINDOW;

    //compute recurrent synapses
    __shared__ float rec_synapses[W_REC_SIZE];
    if(pre < N_REC && post < N_REC) {
        float last_spike = spike_trains[N_REC*last_t + pre];
        rec_synapses[N_REC*pre + post] = last_spike*weight_rec[N_REC*pre + post];
    }
    __syncthreads();

    //compute input synapses
    __shared__ float in_synapses[W_IN_SIZE];
    if(pre < N_IN && post < N_REC) {
        int tn = t%N_WINDOW;
        float current_current = in_currents[N_IN*tn + pre];
        in_synapses[N_REC*pre + post] = current_current*weight_in[N_REC*pre + post];
    }
    __syncthreads();

    //printf("%d  %d  %f\n", row, col, elmntwise_mul[N_REC*row + col]);

    //compute new voltages and spikes
    if(pre == 0 && post < N_REC) {

        int tm = t % N_WINDOW;
        int index = N_REC*tm + post;
        int last_index = N_REC*last_t + post;

        //if a spike occurred in the last step, or we are in the refractory period, clamp the voltage and spike trains to 0
        if(spike_trains[last_index] > 0.5 || refractory_periods[last_index] > 0) {
            voltages[index] = 0.0;
            spike_trains[index] = 0.0;
            refractory_periods[index] = (1 + refractory_periods[N_REC*last_t + post])%REF_PERIOD;
        }

        //otherwise sum the synaptic potentials and possibly generate a spike
        else {

            voltages[index] = VOLT_COEFF*voltages[N_REC*last_t + post];

            for(size_t pre_ = 0; pre_ < N_REC; ++pre_) {
                voltages[index] += rec_synapses[N_REC*pre_ + post];
            }

            for(size_t pre_ = 0; pre_ < N_IN; ++pre_) {
                voltages[index] += in_synapses[N_REC*pre_ + post];
            }

            spike_trains[index] = voltages[index] > THRESHOLD ? 1.0 : 0.0;
        }
    }
}

__global__ void stepLIF(float* voltages,
                        float* spike_trains,
                        int* refractory_buffer,
                        float* in_currents,
                        float* weights_in,
                        float* weights_rec,
                        int t) {

    int post = blockIdx.x*blockDim.x + threadIdx.x;

    //compute new voltages and spikes
    if(post < N_REC) {

        int last_t = (t - 1)%N_WINDOW;
        int tm = t%N_WINDOW;
        int index = N_REC*tm + post;
        int last_index = N_REC*last_t + post;

        //if a spike occurred in the last step, or we are in the refractory period, clamp the voltage and spike trains to 0
        if(spike_trains[last_index] > 0.5 || refractory_buffer[last_index] > 0) {
            voltages[index] = 0.0;
            spike_trains[index] = 0.0;
            refractory_buffer[index] = (1 + refractory_buffer[last_index])%REF_PERIOD;
        }

        //otherwise sum the synaptic potentials and possibly generate a spike
        else {

            voltages[index] = VOLT_COEFF*voltages[last_index];

            //recurrent
            for(size_t pre_ = 0; pre_ < N_REC; ++pre_) {
                voltages[index] += weights_rec[N_REC*pre_ + post]*spike_trains[N_REC*last_t + pre_]; 
            }

            //input
            for(size_t pre_ = 0; pre_ < N_IN; ++pre_) {
                voltages[index] += weights_in[N_REC*pre_ + post]*in_currents[N_IN*last_t + pre_];
            }

            spike_trains[index] = voltages[index] > THRESHOLD ? 1.0 : 0.0;
        }
    }
}

int main() {

    float w_rec[W_REC_SIZE];
    for(size_t i = 0; i < N_REC; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            w_rec[N_REC*i + j] = uniform(-1.0, 1.0);
        }
    //printf("\n");
    }
    
    float w_in[W_IN_SIZE];
    for(size_t i = 0; i < N_IN; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            w_in[N_REC*i + j] = uniform(0.0, 1.0);
        }
    //printf("\n");
    }

    float volts[NET_STATE_SIZE];
    for(size_t i = 0; i < N_WINDOW; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            volts[N_REC*i + j] = uniform(-1.0, 1.0);
        }
    }

    float spikes[NET_STATE_SIZE];
    for(size_t i = 0; i < N_WINDOW; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            spikes[N_REC*i + j] = 0.0;
        }
    }

    float in_currents[N_WINDOW*N_IN];
    for(size_t i = 0; i < N_WINDOW; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            in_currents[N_REC*i + j] = uniform(0.0, 0.3);
        }
    }

    int ref_periods[NET_STATE_SIZE];
    for(size_t i = 0; i < N_WINDOW; ++i) {
        for(size_t j = 0; j < N_REC; ++j) {
            ref_periods[N_REC*i + j] = 0;
        }
    }

    //printf("\n");
    float* w_in_gpu;
    float* w_rec_gpu;
    float* volts_gpu;
    float* spikes_gpu;
    int* ref_periods_gpu;
    float* in_currents_gpu; //pre-determined current so that we don't have to genereate it every time step

    cudaMalloc(&w_in_gpu, W_IN_SIZE*sizeof(float));
    cudaMalloc(&w_rec_gpu, W_REC_SIZE*sizeof(float));
    cudaMalloc(&volts_gpu, NET_STATE_SIZE*sizeof(float));
    cudaMalloc(&spikes_gpu, NET_STATE_SIZE*sizeof(float));
    cudaMalloc(&ref_periods_gpu, NET_STATE_SIZE*sizeof(int));
    cudaMalloc(&in_currents_gpu, N_IN*N_WINDOW*sizeof(float));

    cudaMemcpy(w_in_gpu, w_in, W_IN_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_rec_gpu, w_rec, W_REC_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(volts_gpu, volts, NET_STATE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spikes_gpu, spikes, NET_STATE_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ref_periods_gpu, ref_periods, NET_STATE_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_currents_gpu, in_currents, N_WINDOW*N_IN*sizeof(float), cudaMemcpyHostToDevice);

    for(int time = 1; time < TIMESTEPS; ++time) {

        
        if(time%N_WINDOW == 0) {
            for(size_t i = 0; i < N_WINDOW; ++i) {
                for(size_t j = 0; j < N_REC; ++j) {
                    in_currents[N_REC*i + j] = 0.0;//uniform(0.0, 1.0);
                }
            }
            cudaMemcpy(in_currents_gpu, in_currents, N_WINDOW*N_IN*sizeof(float), cudaMemcpyHostToDevice);
        }
        

        stepLIF<<<4, N_REC/3>>>(volts_gpu, spikes_gpu, ref_periods_gpu, in_currents_gpu, w_in_gpu, w_rec_gpu, time);
    }

    cudaMemcpy(w_in, w_in_gpu, W_IN_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_rec, w_rec_gpu, W_REC_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(volts, volts_gpu, NET_STATE_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(spikes, spikes_gpu, NET_STATE_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_periods, ref_periods_gpu, NET_STATE_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(w_in_gpu);
    cudaFree(w_rec_gpu);
    cudaFree(volts_gpu);
    cudaFree(spikes_gpu);
    cudaFree(ref_periods_gpu);
    cudaFree(in_currents_gpu);

    for(size_t i = 0; i < N_WINDOW; ++i) {
        //printf("%2d", i);
        cout << setw(5) << i;
        for(size_t j = 0; j < N_REC; ++j) {
            //printf("  ");
            //printf("%4.2f", volts[N_REC*i + j]);
            cout << "  ";
            cout << fixed << setprecision(2) << setw(5) << volts[N_REC*i + j];
        }
        //printf("    ");
        cout << "    ";
        for(size_t j = 0; j < N_REC; ++j) {
            //printf("  ");
            //printf("%4.2f", spikes[N_REC*i + j]);
            cout << "  ";
            cout << fixed << setprecision(2) << setw(5) << spikes[N_REC*i + j];
        }
        cout << endl;
    }

    return 0;

}
