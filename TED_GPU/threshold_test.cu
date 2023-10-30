#include "TED_C++.h"

__device__ void task_GPU_threshold_test(int v, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int thread_in_number, int table_in_number, int L, int m, int n){
    int row = table_in_number / L;
    int column = table_in_number % L;
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index;
    int i;
    int j;

//    int i_0 = x_kr[row];
//    int j_0 = y_kr[column];
//    int i_max = x_orl[i_0] + 1;
//    int j_max = y_orl[j_0] + 1;
    int i_0 = 0;
    int j_0 = 0;
    int i_max = v;
    int j_max = v;
//    printf("i_0 = %d, j_0 = %d, i_max = %d, j_max = %d\n", i_0, j_0, i_max, j_max);

    D[j_max + i_max*width + num_table_offset] = 0;

    for (i = i_max - 1; i > i_0 - 1; i--) {
        index = j_max + i*width + num_table_offset;
        D[index] = 1 + D[index + width];
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
        index = j + i_max*width + num_table_offset;
        D[index] = 1 + D[index + 1];
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {
            if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

                index = j + i*width + num_table_offset;

                D[index] = min3_D(Delta[j+i*n] + D[index + 1 + width],
                                  1 + D[index + width],
                                  1 + D[index + 1]);
                D_tree[j + i*n] = D[index];

            } else {

                index = j + i*width + num_table_offset;
                D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                                  1 + D[index + width],
                                  1 + D[index + 1]);

            }
        }
    }
}

__global__ void simple_parallel_threshold_test(int v, int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x < limitation) {
        for (int i = (threadIdx.x + blockIdx.x * blockDim.x); i < d_queue.size(); i += limitation) {
            int thread_in_number = i % (limitation);
            task_GPU_threshold_test(v, x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, thread_in_number, d_queue[i], L, m, n);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multi_table_parallel_warp_28_threshold_test(int* p_d_column_major, int* unit_per_threads, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;

    if(warp_id <2048) {
//        for (int i = warp_id; i < size; i += (gridDim.x * blockDim.x) / 32) {
        for (int i = warp_id; i < size; i += 2048) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];

//            if(blockIdx.x == 0 && threadIdx.x == 0){
//                printf("i_0 = %d, j_0 = %d, i_max = %d, j_max = %d\n", i_0, j_0, i_max, j_max);
//            }
            int x_begin, y_begin, y_end, x_end;
            int iteration;
            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max - 32 * k;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 31;
                    } else {
                        x_end = i_0;
                    }

                    int x_loc = x_begin - lane_id;
                    int y_loc = y_begin;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            y_loc--;
                        }
                        __syncwarp();
                    }

                }


            } else {
                iteration = (j_max - j_0 + 32) / 32;

                for (int k = 0; k < iteration; k++) {

                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 32 * k;

                    if (k < iteration - 1) {
                        y_end = y_begin - 31;
                    } else {
                        y_end = j_0;
                    }

                    int x_loc = x_begin;
                    int y_loc = y_begin - lane_id;

                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, warp_id, m, n);
                            x_loc--;
                        }
                        __syncwarp();
                    }


                }

            }

        }
    }
}



void multi_table_warp_28_threshold_test (int v, Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);
    vector<int> unit_per_threads (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
//        i_0_array[i] = x_kr[row];
//        j_0_array[i] = y_kr[column];
//        i_max_array[i] = x_orl[i_0_array[i]] + 1;
//        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        i_0_array[i] = 0;
        j_0_array[i] = 0;
        i_max_array[i] = v;
        j_max_array[i] = v;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            unit_per_threads[i] = j_max_array[i]-j_0_array[i]+1;
            column_major[i] = 1;
        }else{
            unit_per_threads[i] = i_max_array[i]-i_0_array[i]+1;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d unit_per_thread = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), unit_per_threads[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);
    thrust::device_vector<int> d_unit_per_threads(unit_per_threads);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());
    int *p_d_unit_per_threads = thrust::raw_pointer_cast(d_unit_per_threads.data());


    int block = 256;
    int grid = 256;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

//    printf("--- TTTTTTT WARP METHOD: %d \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    multi_table_parallel_warp_28_threshold_test<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_unit_per_threads, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("%.3f\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void multi_table_parallel_new_update_8_threshold_test(int* p_d_column_major, int* unit_per_threads, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){

//    if(blockIdx.x == 0 && threadIdx.x == 0){
//        printf("table no. %d is computed\n", 0);
//    }

    for (int i=blockIdx.x;i<size; i+=gridDim.x){
        int i_max = p_i_max[i];
        int j_max = p_j_max[i];
        int i_0 = p_i_0[i];
        int j_0 = p_j_0[i];

//        if(blockIdx.x == 0 && threadIdx.x == 0){
//            printf("i_0 = %d, j_0 = %d, i_max = %d, j_max = %d\n", i_0, j_0, i_max, j_max);
//        }
        int x_loc, y_loc;
        int num_in_thread = i % 2143;

        if(p_d_column_major[i] == 1){
            x_loc = i_max - threadIdx.x;
            y_loc = j_max;
        }else{
            x_loc = i_max;
            y_loc = j_max - threadIdx.x;
        }

        int k=0;
        for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
            if((x_loc + y_loc == i_max+j_max - step) && (k<unit_per_threads[i]) && y_loc>=j_0 && x_loc>=i_0 ){
                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                               Delta, D, D_tree, num_in_thread, m, n);
                if(p_d_column_major[i] == 1){
                    y_loc--;
                }else{
                    x_loc--;
                }
                k++;
            }
            __syncthreads();
        }
    }

}


void multi_table_new_update_8_threshold_test (int v, Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);
    vector<int> unit_per_threads (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
//        i_0_array[i] = x_kr[row];
//        j_0_array[i] = y_kr[column];
//        i_max_array[i] = x_orl[i_0_array[i]] + 1;
//        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        i_0_array[i] = 0;
        j_0_array[i] = 0;
        i_max_array[i] = v;
        j_max_array[i] = v;



        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            unit_per_threads[i] = j_max_array[i]-j_0_array[i]+1;
            column_major[i] = 1;
        }else{
            unit_per_threads[i] = i_max_array[i]-i_0_array[i]+1;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d unit_per_thread = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), unit_per_threads[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);
    thrust::device_vector<int> d_unit_per_threads(unit_per_threads);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());
    int *p_d_unit_per_threads = thrust::raw_pointer_cast(d_unit_per_threads.data());


    int block = 1024;
    int grid = 256;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

//    printf("--- TTTTTT BLOCK METHOD: %u\n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

//    multi_table_parallel_new_update<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
    multi_table_parallel_new_update_8_threshold_test<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_unit_per_threads, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%.3f\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ volatile int Array_in_2[82];
__device__ volatile int Array_out_2[82];

__global__ void multi_table_parallel_new_update_2_test_threshold(int* p_d_column_major, int* blocks_per_table, int size, int unit_per_thread, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int tile_size, int* Delta, int* D, int* D_tree, int n, int m){
    int block_begin = 0;
    int block_end = blocks_per_table[0]-1;

//    int block_begin = 0;
//    int block_end = 63;


    for (int i=0; i< size; i++){

        if(blockIdx.x<= block_end & blockIdx.x >= block_begin){

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
//            if(blockIdx.x == 0 && threadIdx.x == 0){
//                printf("i_0 = %d, j_0 = %d, i_max = %d, j_max = %d\n", i_0, j_0, i_max, j_max);
//            }
            int x_loc, y_loc;

            if(p_d_column_major[i] == 1){
                x_loc = i_max - threadIdx.x;
                y_loc = j_max - unit_per_thread*(blockIdx.x-block_begin);
            }else{
                x_loc = i_max - unit_per_thread*(blockIdx.x-block_begin);
                y_loc = j_max - threadIdx.x;
            }
            int k=0;
            for (int step = 0; step < i_max+j_max-i_0-j_0+1; step++) {
                if((x_loc + y_loc == i_max+j_max - step) && (k<unit_per_thread) && y_loc>=j_0 && x_loc>=i_0 ){
                    single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                   Delta, D, D_tree, i, m, n);
                    if(p_d_column_major[i] == 1){
                        y_loc--;
                    }else{
                        x_loc--;
                    }
                    k++;
                }
//                __gpu_sync_range_fence(block_begin,block_end,step+1,Array_in,Array_out);
                __threadfence();
                __gpu_sync_range_update(block_begin,block_end,step+1,Array_in_2,Array_out_2);
            }
        }

        if (block_end + blocks_per_table[i+1] <= 63){
            block_begin = block_end + 1;
            block_end = block_end + blocks_per_table[i+1];
        } else{
            block_begin = 0;
            block_end = blocks_per_table[i+1]-1;
        }
    }

}


void multi_table_new_update_test_threshold (int v, Stream& stream_mul, float& total_milliseconds, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, dev::Queue<int, uint32_t> d_queue, int size, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    int unit_per_thread = 64;

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
//        i_0_array[i] = x_kr[row];
//        j_0_array[i] = y_kr[column];
//        i_max_array[i] = x_orl[i_0_array[i]] + 1;
//        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        i_0_array[i] = 0;
        j_0_array[i] = 0;
        i_max_array[i] = v;
        j_max_array[i] = v;

        blocks_per_table[i] = (j_max_array[i] - j_0_array[i] + unit_per_thread)/unit_per_thread;
        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            blocks_per_table[i] = (j_max_array[i] - j_0_array[i] + unit_per_thread)/unit_per_thread;
            column_major[i] = 1;
        }else{
            blocks_per_table[i] = (i_max_array[i]-i_0_array[i]+ unit_per_thread)/unit_per_thread;
            column_major[i] = 0;
        }
//        printf("--- TableSize = %d NumBlocks = %d row = %d, column = %d \n", (i_max_array[i]-i_0_array[i]+1)*(j_max_array[i]-j_0_array[i]+1), blocks_per_table[i], i_max_array[i]-i_0_array[i]+1, j_max_array[i]-j_0_array[i]+1);
    }

    thrust::device_vector<int> d_i_0_array(i_0_array);
    thrust::device_vector<int> d_i_max_array(i_max_array);
    thrust::device_vector<int> d_j_0_array(j_0_array);
    thrust::device_vector<int> d_j_max_array(j_max_array);
    thrust::device_vector<int> d_blocks_per_table(blocks_per_table);
    thrust::device_vector<int> d_column_major(column_major);

    int *p_d_i_0_array = thrust::raw_pointer_cast(d_i_0_array.data());
    int *p_d_i_max_array = thrust::raw_pointer_cast(d_i_max_array.data());
    int *p_d_j_0_array = thrust::raw_pointer_cast(d_j_0_array.data());
    int *p_d_j_max_array = thrust::raw_pointer_cast(d_j_max_array.data());
    int *p_d_blocks_per_table = thrust::raw_pointer_cast(d_blocks_per_table.data());
    int *p_d_column_major = thrust::raw_pointer_cast(d_column_major.data());

    int block = 1024;
    int grid = 64;
    int tile = 1;
    int block_per_table = grid / queue_h.size();

//    printf("----- TTTTTTTT MANY BLOCKS: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

//    multi_table_parallel_new_update<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);
    multi_table_parallel_new_update_2_test_threshold<<<grid, block, 0, stream_mul.cuda_stream()>>>(p_d_column_major, p_d_blocks_per_table, queue_h.size(), unit_per_thread, p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, tile, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%.3f\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


vector<vector<int>> parallel_standard_ted_test_threshold(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();


    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();


    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

// Preprocessing

    for (int i=0;i<K;i++){
        int node = x_kr[i];
        if((node == x_orl[node])||(node == x_orl[node]-1)){
            x_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = x_kr[j];
                if ((node <= node_2)&&(x_orl[node] >= node_2)){
                    x_keyroot_depth_2[i] = max(x_keyroot_depth_2[i],x_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    for (int i=0;i<L;i++){
        int node = y_kr[i];
        if((node == y_orl[node])||(node == y_orl[node]-1)){
            y_keyroot_depth_2[i] = 0;
        }else{
            for (int j=0;j<i;j++){
                int node_2 = y_kr[j];
                if ((node <= node_2)&&(y_orl[node] >= node_2)){
                    y_keyroot_depth_2[i] = max(y_keyroot_depth_2[i],y_keyroot_depth_2[j]+1);
                }
            }
        }
    }

    int max_depth = 0;
    for (int i=0;i<K*L;i++){
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
        if(depth[i] > max_depth){
            max_depth = depth[i];
        }
    }
    printf("max_depth = %d\n", max_depth);


    // GPU Programming
    int blockSize = 256;
    int current_depth = 0;
    double limitation = (8*1024*1024*1024.0)/((m+1)*(n+1)*4);
    printf("limitation is %u\n", int(limitation));
    int batch_size = min(K*L,int(limitation));

    // x_orl, x_kr, y_orl, y_kr
    thrust::device_vector<int> x_orl_d(x_orl);
    thrust::device_vector<int> x_kr_d(x_kr);
    thrust::device_vector<int> y_orl_d(y_orl);
    thrust::device_vector<int> y_kr_d(y_kr);

    // Delta, D_tree
    vector<int> Delta_trans (m*n);
    for (int i=0; i<m;i++){
        for (int j=0; j<n;j++){
            Delta_trans[i*n+j] = Delta[i][j];
        }
    }
    thrust::device_vector<int> Delta_d(Delta_trans);
    vector<int> D_tree_trans(m*n, 0);
    thrust::device_vector<int> D_tree_d(D_tree_trans);

    // depth
    thrust::device_vector<int> depth_d(depth);

    // D
//    vector<int> D_trans((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d(D_trans);
    thrust::device_vector<int> D_d((m+1)*(n+1)*batch_size, 0);
//    thrust::device_vector<int> D_d_2((m+1)*(n+1)*batch_size, 0);

    // Size
//    vector<int> size(K*L,0);
//    thrust::device_vector<int> size_d(size);

    // Pointer: x_orl, x_kr, y_orl, y_kr, Delta, D_tree, depth, worklist1, worklist2, D
    int *p_x_orl_d = thrust::raw_pointer_cast(x_orl_d.data());
    int *p_x_kr_d = thrust::raw_pointer_cast(x_kr_d.data());
    int *p_y_orl_d = thrust::raw_pointer_cast(y_orl_d.data());
    int *p_y_kr_d = thrust::raw_pointer_cast(y_kr_d.data());
    int *p_Delta_d = thrust::raw_pointer_cast(Delta_d.data());
    int *p_D_tree_d = thrust::raw_pointer_cast(D_tree_d.data());
    int *p_depth_d = thrust::raw_pointer_cast(depth_d.data());
    int *p_D_d = thrust::raw_pointer_cast(D_d.data());
//    int *p_D_d_2 = thrust::raw_pointer_cast(D_d_2.data());

    float total_milliseconds = 0;

    Stream stream;

    ArrayView<int> depth_d_view(depth_d);

    Queue<int> queue1;
    queue1.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue1 = queue1.DeviceObject();

    Queue<int> queue2;
    queue2.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_queue2 = queue2.DeviceObject();

    Queue<int> large_queue;
    large_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_large_queue = large_queue.DeviceObject();
    large_queue.set_size(stream, 0);

    Queue<int> block_size_queue;
    block_size_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_block_size_queue = block_size_queue.DeviceObject();
    block_size_queue.set_size(stream, 0);


    // Fetch task

    int numBlocks = (int)(depth_d_view.size() + blockSize - 1) / blockSize;
    fetch_task<<<numBlocks,blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, 0);
    stream.Sync();

    initializeArrayToZero<<<1, 64>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }



    while (queue1.size() !=0 || large_queue.size() != 0 || block_size_queue.size() != 0) {

        printf("\ncurrent depth = %d, batch_size = %d\n", current_depth, batch_size);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (current_depth == 7){


            int num_task = queue1.size();
            numBlocks = (num_task + blockSize - 1) / blockSize;
            numBlocks = 128;

            printf("---------------------Test\n");

            printf("\n");
            printf("\n");

            queue1.set_size(stream, 1);

            printf("Single\n");
            for (int v=1; v<15; v+=1) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

//                printf("%d\n", (v + 1));

                cudaEventRecord(start, stream.cuda_stream());
                if (queue1.size() > 0) {
//                    printf("--- TTTTTTTTTT SINGLE METHOD: %d\n", queue1.size());
                    simple_parallel_threshold_test<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(v, p_x_orl_d,
                                                                                                      p_x_kr_d,
                                                                                                      p_y_orl_d,
                                                                                                      p_y_kr_d,
                                                                                                      p_Delta_d, p_D_d,
                                                                                                      p_D_tree_d, n, m,
                                                                                                      L, d_queue1,
                                                                                                      batch_size);
                }

                cudaEventRecord(stop, stream.cuda_stream());
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
//                printf("%.3f\n", milliseconds);
                total_milliseconds += milliseconds;
            }

            printf("\n");
            printf("\n");
            printf("Warp\n");

            for (int v=78; v<105; v+=1) {
//                printf("size = %d ", (v + 1) * (v + 1));
                if (queue1.size() > 0) {
                    thrust::host_vector<int> large_queue_h(queue1.size());
                    cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()), queue1.data(),
                                    queue1.size() * sizeof(int), cudaMemcpyDeviceToHost, stream.cuda_stream());
                    stream.Sync();
                    multi_table_warp_28_threshold_test(v, stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl,
                                                       y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d,
                                                       p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
                }
            }

                printf("\n");
                printf("\n");

            printf("Block\n");

            for (int v=900; v<1010; v+=10) {
//                printf("size = %d ", (v + 1) * (v + 1));

                if (queue1.size() > 0) {
                    thrust::host_vector<int> block_size_queue_h(queue1.size());
                    cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()), queue1.data(),
                                    queue1.size() * sizeof(int), cudaMemcpyDeviceToHost, stream.cuda_stream());
                    stream.Sync();
                    multi_table_new_update_8_threshold_test(v, stream, total_milliseconds, block_size_queue_h, x_orl,
                                                            x_kr,
                                                            y_orl, y_kr, d_queue1, block_size_queue_h.size(), L, n, m,
                                                            p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
                }
            }
                printf("\n");
                printf("\n");
            printf("Many Blocks\n");

            for (int v=900; v<1010; v+=10) {
//                printf("size = %d ", (v + 1) * (v + 1));
                if (queue1.size() > 0) {
                    thrust::host_vector<int> huge_size_queue_h(queue1.size());
                    cudaMemcpyAsync(thrust::raw_pointer_cast(huge_size_queue_h.data()), queue1.data(),
                                    queue1.size() * sizeof(int), cudaMemcpyDeviceToHost, stream.cuda_stream());
                    stream.Sync();
                    multi_table_new_update_test_threshold(v, stream, total_milliseconds, huge_size_queue_h, x_orl, x_kr,
                                                          y_orl, y_kr, d_queue1, huge_size_queue_h.size(), L, n, m,
                                                          p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
                }
            }
            printf("\n");
            printf("\n");

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (queue1.size()<=5 && queue1.size()>6){

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else if (current_depth == max_depth){

            cudaEvent_t start, stop;
            float milliseconds = 0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, stream.cuda_stream());


            single_table_new_9(stream, K*L-1, 0, x_orl, x_kr, y_orl, y_kr, L, blockSize, n, m , p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("Measured time for parallel execution = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        }else{
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);


            int num_task = queue1.size();
            numBlocks = (num_task + blockSize - 1) / blockSize;
            numBlocks = 128;
//            printf("numBlocks = %d，Hello 28 numTasks = %d\n", numBlocks,num_task);

//            thrust::host_vector<int> queue_h(queue1.size());
//            cudaMemcpyAsync(thrust::raw_pointer_cast(queue_h.data()),
//                            queue1.data(), queue1.size() * sizeof(int),
//                            cudaMemcpyDeviceToHost, stream.cuda_stream());
//            stream.Sync();


            cudaEventRecord(start, stream.cuda_stream());
            if(queue1.size()>0){
                printf("--- SINGLE METHOD: %d\n", queue1.size());
                simple_parallel_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, batch_size);
            }

            cudaEventRecord(stop, stream.cuda_stream());
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("Measured time for SINGLE = %.3fms\n", milliseconds);
            total_milliseconds+=milliseconds;

            if(large_queue.size() != 0){
                thrust::host_vector<int> large_queue_h(large_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()),large_queue.data(), large_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_table_warp_28(stream, total_milliseconds, large_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, large_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

            }

            if(block_size_queue.size() != 0){
                thrust::host_vector<int> block_size_queue_h(block_size_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()),block_size_queue.data(), block_size_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_table_new_update_8(stream, total_milliseconds, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, d_queue1, block_size_queue_h.size(), L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

            }

        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        queue1.set_size(stream, 0);
        large_queue.set_size(stream, 0);
        block_size_queue.set_size(stream, 0);
        current_depth++;
        numBlocks = (depth_d_view.size() + blockSize - 1) / blockSize;
        fetch_task<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, current_depth);
        stream.Sync();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Sort by Size
        queue2.set_size(stream, queue1.size());
        numBlocks = (queue1.size() + blockSize - 1) / blockSize;

        fetch_size_queue<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue2, d_queue1,p_x_orl_d, p_y_orl_d, p_x_kr_d, p_y_kr_d, L);
        stream.Sync();

        thrust::sort_by_key(queue2.getRawPointer(), queue2.getRawPointer()+queue1.size(), queue1.getRawPointer());
        cudaStreamSynchronize(stream.cuda_stream());
        queue2.set_size(stream,0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//        if(current_depth < 6 && current_depth >4) {
//            printQueueVector<<<1, 1, 0, stream.cuda_stream()>>>(d_queue1);
        numBlocks = (queue1.size() + blockSize - 1) / blockSize;
        filter<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, 1000000, 1000000);
//            filter<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue,15, 10000000);
        stream.Sync();
        int old_size = queue1.size();
        queue1.set_size(stream,old_size - large_queue.size()-block_size_queue.size());

        printf("//////////////////////////////////\n");
        printf("QUEUE1.size = %d\n", queue1.size());
        printf("Large_queue.size = %d\n", large_queue.size());
        printf("block_size_queue.size = %d\n", block_size_queue.size());
//        }
    }


    int final = D_tree_d[0];
    printf("The total final distance is %u\n", final);
    printf("hello\n");
    printf("Measured time for total parallel execution = %.3fms\n", total_milliseconds);

    cudaMemcpy(D_tree_trans.data(), p_D_tree_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("D[0][13] = %d, D[61][0] = %d\n", D_tree_trans[0*1000+13], D_tree_trans[61*1000+0]);

    vector<vector<int>> final_result(m,vector<int>(n,0));
    return final_result;

}


