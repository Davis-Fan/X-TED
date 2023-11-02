#include "TED_C++.h"

//__device__ int min3_D(const int& a, const int& b, const int& c){
__device__ int min3_D(int a, int b, int c){
//    int min_val = a;
//    if (b < min_val) {
//        min_val = b;
//    }
//    if (c < min_val) {
//        min_val = c;
//    }
//    return min_val;
    return min(min(a,b),c);
}


// Fetch task according to their depths
__global__ void fetch_task(ArrayView<int> d_view, dev::Queue<int, uint32_t> d_queue, int current_depth){
    for(int i=TID_1D;i<d_view.size();i+=TOTAL_THREADS_1D) {
        if (d_view[i] == current_depth){
            d_queue.Append(i);
        }
    }
}


// Compute the size of each table
__global__ void fetch_size_queue(dev::Queue<int, uint32_t> d_size, dev::Queue<int, uint32_t> d_queue, int* x_orl, int* y_orl, int* x_kr, int* y_kr, int L){
    for (int i=TID_1D;i<d_queue.size();i+=TOTAL_THREADS_1D){
        int task = d_queue[i];
        int row = task / L;
        int column = task % L;
        int i_0 = x_kr[row];
        int j_0 = y_kr[column];
        d_size[i] =  (x_orl[i_0] + 1-i_0+1)*(y_orl[j_0] + 1-j_0+1);
    }
}


// Categorize tables into four size types
__global__ void filter_new(dev::Queue<int, uint32_t> d_queue1, dev::Queue<int, uint32_t> d_queue2, dev::Queue<int, uint32_t> large_queue, dev::Queue<int, uint32_t> block_size_queue, dev::Queue<int, uint32_t> multi_block_queue, int threshold1, int threshold2, int threshold3){
    for (int i=TID_1D;i<d_queue1.size();i+=TOTAL_THREADS_1D){
        if (d_queue2[i] >= threshold1 && d_queue2[i] < threshold2){
            int task = d_queue1[i];
            large_queue.Append(task);
        }

        if(d_queue2[i] >= threshold2 && d_queue2[i] < threshold3){
            block_size_queue.Append(d_queue1[i]);
        }

        if(d_queue2[i] >= threshold3){
            multi_block_queue.Append(d_queue1[i]);
        }

    }
}

