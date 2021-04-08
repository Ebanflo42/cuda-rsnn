#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace cudakernel {

__global__ void neuronStep(float* voltages,
                           float* spikeTrains,
                           int* refractoryPeriods,
                           float* inCurrents,
                           float* weightIn,
                           float* weightRec,
                           int nIn,
                           int nRec,
                           int wInSize,
                           int wRecSize,
                           int window,
                           int refPeriod,
                           float voltCoeff,
                           float threshold,
                           int t) {

    int pre = blockIdx.x*blockDim.x + threadIdx.x;
    int post = blockIdx.y*blockDim.y + threadIdx.y;

    int last_t = (t - 1)%window;

    //compute recurrent synapses
    __shared__ float rec_synapses[1000];
    if(pre < nRec && post < nRec) {
        float last_spike = spikeTrains[nRec*last_t + pre];
        rec_synapses[nRec*pre + post] = last_spike*weightRec[nRec*pre + post];
    }
    __syncthreads();

    //compute input synapses
    __shared__ float in_synapses[1000];
    if(pre < nIn && post < nRec) {
        int tn = t%window;
        float current_current = inCurrents[nIn*tn + pre];
        in_synapses[nRec*pre + post] = current_current*weightIn[nRec*pre + post];
    }
    __syncthreads();

    //printf("%d  %d  %f\n", row, col, elmntwise_mul[nRec*row + col]);

    //compute new voltages and spikes
    if(pre == 0 && post < nRec) {

        int tm = t%window;
        int index = nRec*tm + post;
        int last_index = nRec*last_t + post;

        //if a spike occurred in the last step, or we are in the refractory period, clamp the voltage and spike trains to 0
        if(spikeTrains[last_index] > 0.5 || refractoryPeriods[last_index] > 0) {
            voltages[index] = 0.0;
            spikeTrains[index] = 0.0;
            refractoryPeriods[index] = (1 + refractoryPeriods[nRec*last_t + post])%refPeriod;
        }

        //otherwise sum the synaptic potentials and possibly generate a spike
        else {

            voltages[index] = voltCoeff*voltages[last_index];

            for(size_t pre_ = 0; pre_ < nRec; ++pre_) {
                voltages[index] += rec_synapses[nRec*pre_ + post];
            }

            for(size_t pre_ = 0; pre_ < nIn; ++pre_) {
                voltages[index] += in_synapses[nRec*pre_ + post];
            }

            spikeTrains[index] = voltages[index] > threshold ? 1.0 : 0.0;
        }
    }
}

}
