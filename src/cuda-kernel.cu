#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace cudakernel {

__global__ void stepLIF(float* voltages,
                        float* spike_trains,
                        int* refractory_buffer,
                        float* in_currents,
                        float* weights_in,
                        float* weights_rec,
                        int n_in,
                        int n_rec,
                        int weights_in_size,
                        int weights_rec_size,
                        int buffer_len,
                        int ref_period,
                        float volt_coeff,
                        float threshold,
                        int start,
                        int steps) {

    int post = blockIdx.x*blockDim.x + threadIdx.x;

    //compute new voltages and spikes
    if(post < n_rec) {

        //printf("%d", sizeof(spike_trains)/sizeof(float));
        int end = start + steps;

        for(int t = start; t < end; ++t) {

            int last_t = t == 0 ? buffer_len - 1 : t - 1;
            int tm = t%buffer_len;
            int index = n_rec*tm + post;
            int last_index = n_rec*last_t + post;

            //if a spike occurred in the last step, or we are in the refractory period, clamp the voltage and spike trains to 0
            if(spike_trains[last_index] > 0.5 || refractory_buffer[last_index] > 0) {
                //printf("spike\n");
                voltages[index] = 0.0;
                spike_trains[index] = 0.0;
                refractory_buffer[index] = (1 + refractory_buffer[last_index])%ref_period;
            }

            //otherwise sum the synaptic potentials and possibly generate a spike
            else {

                /*
                if(t < 20) {
                    printf("%d %f\n", post, voltages[index]);
                }
                */

                voltages[index] = volt_coeff*voltages[last_index];

                //recurrent
                for(size_t pre_ = 0; pre_ < n_rec; ++pre_) {
                    voltages[index] += weights_rec[n_rec*pre_ + post]*spike_trains[n_rec*last_t + pre_];
                }

                //input
                for(size_t pre_ = 0; pre_ < n_in; ++pre_) {
                    voltages[index] += weights_in[n_rec*pre_ + post]*in_currents[n_in*last_t + pre_];
                }

                spike_trains[index] = voltages[index] > threshold ? 1.0 : 0.0;
            }
            __syncthreads();
        }
    }
}

}
