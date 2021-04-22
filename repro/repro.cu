#include <stdio.h>

float uniform(float low, float high) {
    return low + (static_cast<float>(rand())/RAND_MAX)*(high - low);
}

int main() {

    int size = static_cast<int>(1 << 25);

    float* host_arr = (float*) malloc(size*sizeof(float));
    for(size_t i = 0; i < size; ++i) {
        host_arr[i] = uniform(0.0, 1.0);
    }
    float* dev_arr;

    cudaMalloc(&dev_arr, size*sizeof(float));
    cudaMemcpy(dev_arr, host_arr, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(host_arr, dev_arr, size*sizeof(float), cudaMemcpyDeviceToHost);

    int ix = 4000000;
    printf("%f\n", host_arr[ix]);

    return 0;
}
