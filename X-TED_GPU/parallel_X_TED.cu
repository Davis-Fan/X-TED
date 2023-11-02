#include "TED_C++.h"


// Compute one single table in parallel
__device__ void task_GPU(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int thread_in_number, int table_in_number, int L, int m, int n){
    int row = table_in_number / L;
    int column = table_in_number % L;
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index;
    int i;
    int j;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;

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


// Inter-block lock-free synchronization
__device__ volatile int Array_in[82];
__device__ volatile int Array_out[82];


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ int units_per_thread;

// Sync all threads
__device__ void __gpu_sync(int goalVal, volatile int *Arrayin, volatile int *Arrayout){
    // Thread ID in a block
    int tid_in_blk = threadIdx.x;
    int nBlockNum = gridDim.x;
    int bid = blockIdx.x;

    // Only thread 0 is used for synchronization
    if (tid_in_blk == 0){Arrayin[bid] = goalVal;}

    if (bid == 1){
        if (tid_in_blk < nBlockNum){
            while (Arrayin[tid_in_blk] != goalVal){}
        }
        __syncthreads();
        if (tid_in_blk < nBlockNum){
            Arrayout[tid_in_blk] = goalVal;
        }
    }

    if (tid_in_blk == 0){
        while (Arrayout[bid] != goalVal){}
    }
    __syncthreads();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute one single unit
__device__ void single_unit_10(int* x_orl, int* y_orl, int i_0, int i_max, int j_0, int j_max, int i, int j, int* Delta, int* D, int* D_tree, int thread_in_number, int m, int n){
    int num_table_offset = thread_in_number*(m+1)*(n+1);
    int width = n+1;
    int index = j + i*width + num_table_offset;
//    printf("ok\n");
    if ((i == i_max) & (j == j_max)){
        D[index] = 0;
    }else if ((i <= i_max-1) & (i >= i_0) & (j == j_max)){
        D[index] = 1 + D[index+width];
    }else if ((j <= j_max-1) & (j >= j_0) & (i == i_max)){
        D[index] = 1 + D[1+index];
    }else if ((i <= i_max-1) & (i >= i_0) & (j <= j_max-1) & (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];

        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width + num_table_offset],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Inter-block Synchronization
__device__ void __gpu_sync_range_update(int startBlockIdx, int endBlockIdx, int goalVal, volatile int *Arrayin, volatile int *Arrayout){
    int tid_in_blk = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= startBlockIdx && bid <= endBlockIdx){
        if (tid_in_blk == 0){
                Arrayin[bid] = goalVal;
        }
    }

    __syncthreads();

    if (bid == startBlockIdx){
        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            while (Arrayin[tid_in_blk] != goalVal){}
        }
        __syncthreads();

        if (tid_in_blk >= startBlockIdx & tid_in_blk <= endBlockIdx){
            Arrayout[tid_in_blk] = goalVal;
        }
    }

    if (bid >= startBlockIdx && bid <= endBlockIdx){
        if (tid_in_blk == 0){
                while (Arrayout[bid] != goalVal){}
            }
    }
    __syncthreads();
}


__global__ void initializeArrayToZero() {
    int tid = threadIdx.x;
    if (tid < 64) {
        Array_in[tid] = 0;
        Array_out[tid] = 0;
    }
}


vector<vector<int>> parallel_standard_ted_test(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();


    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();


    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

// Preprocessing
    auto start_time_2 = chrono::steady_clock::now();
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

    auto end_time_2 = chrono::steady_clock::now();
    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
    cout << "Preprocessing Method, the time is " << ms_2/1000.0 << " ms consumed" << endl;

    // GPU Setting
    int blockSize = 256;
    int current_depth = 0;

    // Compute the max number of tables that can be stored in memory
    double limitation = (8*1024*1024*1024.0)/((m+1)*(n+1)*4);
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
    thrust::device_vector<int> D_d((m+1)*(n+1)*batch_size, 0);

    // Pointer: x_orl, x_kr, y_orl, y_kr, Delta, D_tree, depth, worklist1, worklist2, D
    int *p_x_orl_d = thrust::raw_pointer_cast(x_orl_d.data());
    int *p_x_kr_d = thrust::raw_pointer_cast(x_kr_d.data());
    int *p_y_orl_d = thrust::raw_pointer_cast(y_orl_d.data());
    int *p_y_kr_d = thrust::raw_pointer_cast(y_kr_d.data());
    int *p_Delta_d = thrust::raw_pointer_cast(Delta_d.data());
    int *p_D_tree_d = thrust::raw_pointer_cast(D_tree_d.data());
    int *p_depth_d = thrust::raw_pointer_cast(depth_d.data());
    int *p_D_d = thrust::raw_pointer_cast(D_d.data());

    total_milliseconds = 0;

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

    Queue<int> multi_block_queue;
    multi_block_queue.Init(K*L*sizeof(int));
    dev::Queue<int,uint32_t> d_multi_block_queue = multi_block_queue.DeviceObject();
    multi_block_queue.set_size(stream, 0);


    int numBlocks = (int)(depth_d_view.size() + blockSize - 1) / blockSize;
    fetch_task<<<numBlocks,blockSize, 0, stream.cuda_stream()>>>(depth_d_view, d_queue1, 0);
    stream.Sync();

    // Initialize the inter-block synchronization
    initializeArrayToZero<<<1, 64>>>();


    while (queue1.size() !=0 || large_queue.size() != 0 || block_size_queue.size() != 0 || multi_block_queue.size() != 0) {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (current_depth == max_depth){

            last_table(stream, total_milliseconds, K*L-1, 0, x_orl, x_kr, y_orl, y_kr, L, blockSize, n, m , p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);

        }else{

            // Single-Thread Approach
            if(queue1.size()>0) {
                simple_thread(stream, total_milliseconds, limitation, d_queue1, L, n, m, p_x_orl_d, p_y_orl_d, p_x_kr_d,
                              p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

            // Single-Warp Approach
            if(large_queue.size() > 0){
                thrust::host_vector<int> large_queue_h(large_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(large_queue_h.data()),large_queue.data(), large_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                simple_warp(stream, total_milliseconds, limitation, large_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

            // Single-Block Approach
            if(block_size_queue.size() != 0){
                thrust::host_vector<int> block_size_queue_h(block_size_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(block_size_queue_h.data()),block_size_queue.data(), block_size_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                simple_block(stream, total_milliseconds, limitation, block_size_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

            // Multi-Blocks Approach
            if(multi_block_queue.size() != 0){
                thrust::host_vector<int> multi_block_queue_h(multi_block_queue.size());
                cudaMemcpyAsync(thrust::raw_pointer_cast(multi_block_queue_h.data()),multi_block_queue.data(), multi_block_queue.size() * sizeof(int),cudaMemcpyDeviceToHost, stream.cuda_stream());
                stream.Sync();
                multi_block(stream, total_milliseconds, limitation, multi_block_queue_h, x_orl, x_kr, y_orl, y_kr, L, n, m, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d);
            }

        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        queue1.set_size(stream, 0);
        large_queue.set_size(stream, 0);
        block_size_queue.set_size(stream, 0);
        multi_block_queue.set_size(stream, 0);
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
        // Filtering
        numBlocks = (queue1.size() + blockSize - 1) / blockSize;
        filter_new<<<numBlocks, blockSize, 0, stream.cuda_stream()>>>(d_queue1, d_queue2, d_large_queue, d_block_size_queue, d_multi_block_queue, 15, 8750, 4500000);
        stream.Sync();
        int old_size = queue1.size();
        queue1.set_size(stream,old_size -multi_block_queue.size()- large_queue.size()-block_size_queue.size());

    }


    int final = D_tree_d[0];
    printf("The total final distance is %u\n", final);
    printf("Measured time for total parallel execution = %.3fms\n", total_milliseconds);

    cudaMemcpy(D_tree_trans.data(), p_D_tree_d, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    vector<vector<int>> final_result(m,vector<int>(n,0));
    return final_result;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void simple_thread_parallel(int* x_orl, int* x_kr, int* y_orl, int* y_kr, int* Delta, int* D, int* D_tree, int n, int m, int L, dev::Queue<int, uint32_t> d_queue, int limitation){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int total = blockDim.x*gridDim.x;

    if(total>=limitation){
        if(x < limitation) {
            for (int i = (threadIdx.x + blockIdx.x * blockDim.x); i < d_queue.size(); i += limitation) {
                int thread_in_number = i % (limitation);
                task_GPU(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, thread_in_number, d_queue[i], L, m, n);
            }
        }
    }else{
        for (int i = x; i < d_queue.size(); i += total) {
            task_GPU(x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree, x, d_queue[i], L, m, n);
        }
    }
}

void simple_thread(Stream& stream_single, float& total_milliseconds, int limitation, dev::Queue<int, uint32_t> d_queue1, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int*p_x_kr_d, int*p_y_kr_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){

    int block = 256;
    int grid = 128;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_single.cuda_stream());

    simple_thread_parallel<<<grid, block, 0, stream_single.cuda_stream()>>>(p_x_orl_d, p_x_kr_d, p_y_orl_d, p_y_kr_d, p_Delta_d, p_D_d, p_D_tree_d, n, m, L, d_queue1, limitation);

    cudaEventRecord(stop, stream_single.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Single-Thread = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void simple_warp_parallel(int limitation, int* p_d_column_major, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){

    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int lane_id = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    int total = (blockDim.x * gridDim.x)/32;

    if(total <= limitation) {

        for (int i = warp_id; i < size; i += total) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
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

    }else{
        if(warp_id < limitation){
            for (int i = warp_id; i < size; i += limitation) {

                int thread_in_number = i % (limitation);
                int i_max = p_i_max[i];
                int j_max = p_j_max[i];
                int i_0 = p_i_0[i];
                int j_0 = p_j_0[i];
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
                                               Delta, D, D_tree, thread_in_number, m, n);
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
                                               Delta, D, D_tree, thread_in_number, m, n);
                                x_loc--;
                            }
                            __syncwarp();
                        }
                    }
                }

            }
        }
    }
}


void simple_warp (Stream& stream, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            column_major[i] = 1;
        }else{
            column_major[i] = 0;
        }
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


    int block = 256;
    int grid = 256;
    int block_per_table = grid / queue_h.size();

//    printf("--- Warp =: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.cuda_stream());

    simple_warp_parallel<<<grid, block, 0, stream.cuda_stream()>>>(limitation, p_d_column_major, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

//    printf("Single-Warp = %.3fms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    total_milliseconds+=milliseconds;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void simple_block_parallel(int limitation, int* p_d_column_major, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){
    int total = gridDim.x;
    int x = blockIdx.x;

    if(total <= limitation) {

        for (int i = x; i < size; i += total) {

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, x_end, y_end, iteration;

            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + 1024) / 1024;
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max - k * 1024;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - 1023;
                    } else {
                        x_end = i_0;
                    }
                    int x_loc = x_begin - threadIdx.x;
                    int y_loc = y_begin;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, x, m, n);
                            y_loc--;
                        }
                        __syncthreads();
                    }
                }
            } else {
                iteration = (j_max - j_0 + 1024) / 1024;
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - 1024 * k;
                    if (k < iteration - 1) {
                        y_end = y_begin - 1023;
                    } else {
                        y_end = j_0;
                    }
                    int x_loc = x_begin;
                    int y_loc = y_begin - threadIdx.x;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, x, m, n);
                            x_loc--;
                        }
                        __syncthreads();
                    }
                }
            }

        }
    }else{

        if(blockIdx.x < limitation){
            for (int i = x; i < size; i += limitation) {
                int num_in_thread = i % limitation;
                int i_max = p_i_max[i];
                int j_max = p_j_max[i];
                int i_0 = p_i_0[i];
                int j_0 = p_j_0[i];
                int x_begin, y_begin, x_end, y_end, iteration;

                if (p_d_column_major[i] == 1) {
                    iteration = (i_max - i_0 + 1024) / 1024;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max - k * 1024;
                        y_begin = j_max;
                        y_end = j_0;
                        if (k < iteration - 1) {
                            x_end = x_begin - 1023;
                        } else {
                            x_end = i_0;
                        }
                        int x_loc = x_begin - threadIdx.x;
                        int y_loc = y_begin;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {


                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, num_in_thread, m, n);
                                y_loc--;

                            }
                            __syncthreads();
                        }
                    }
                } else {
                    iteration = (j_max - j_0 + 1024) / 1024;
                    for (int k = 0; k < iteration; k++) {
                        x_begin = i_max;
                        x_end = i_0;
                        y_begin = j_max - 1024 * k;
                        if (k < iteration - 1) {
                            y_end = y_begin - 1023;
                        } else {
                            y_end = j_0;
                        }
                        int x_loc = x_begin;
                        int y_loc = y_begin - threadIdx.x;
                        for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                            if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                                single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                               Delta, D, D_tree, num_in_thread, m, n);
                                x_loc--;
                            }
                            __syncthreads();
                        }
                    }
                }
            }
        }
    }
}


void simple_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);

    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;

        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            column_major[i] = 1;
        }else{
            column_major[i] = 0;
        }
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
    int grid = 256;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());

    simple_block_parallel<<<grid, block, 0, stream_mul.cuda_stream()>>>(limitation, p_d_column_major, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void multi_block_parallel(int limitation, int* p_d_column_major, int* blocks_per_table, int size, int* x_orl, int* y_orl, int* p_i_0, int* p_i_max, int* p_j_0, int* p_j_max, int* Delta, int* D, int* D_tree, int n, int m){
    int block_begin = 0;
    int block_end = blocks_per_table[0]-1;

    for (int i=0; i< size; i++){

        if(blockIdx.x<= block_end & blockIdx.x >= block_begin){

            int t = block_end - block_begin + 1;
            int x = threadIdx.x + (blockIdx.x-block_begin)*blockDim.x;
            int num_in_thread = i % limitation;

            int i_max = p_i_max[i];
            int j_max = p_j_max[i];
            int i_0 = p_i_0[i];
            int j_0 = p_j_0[i];
            int x_begin, y_begin, x_end, y_end, iteration;

            if (p_d_column_major[i] == 1) {
                iteration = (i_max - i_0 + t*1024) / (t*1024);
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max - k * t*1024;
                    y_begin = j_max;
                    y_end = j_0;
                    if (k < iteration - 1) {
                        x_end = x_begin - t*1024+1;
                    } else {
                        x_end = i_0;
                    }
                    int x_loc = x_begin - x;
                    int y_loc = y_begin;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, num_in_thread, m, n);
                            y_loc--;
                        }
                        __threadfence();
                        // Lock-free based inter-blocks Synchronization
                        __gpu_sync_range_update(block_begin,block_end,step+1,Array_in,Array_out);
                    }
                }
            } else {
                iteration = (j_max - j_0 + t*1024) / (t*1024);
                for (int k = 0; k < iteration; k++) {
                    x_begin = i_max;
                    x_end = i_0;
                    y_begin = j_max - t*1024 * k;
                    if (k < iteration - 1) {
                        y_end = y_begin - t*1024+1;
                    } else {
                        y_end = j_0;
                    }
                    int x_loc = x_begin;
                    int y_loc = y_begin - x;
                    for (int step = 0; step < x_begin + y_begin - x_end - y_end + 1; step++) {
                        if ((x_loc + y_loc == x_begin + y_begin - step) & (y_loc >= y_end) & (x_loc >= x_end)) {
                            single_unit_10(x_orl, y_orl, i_0, i_max, j_0, j_max, x_loc, y_loc,
                                           Delta, D, D_tree, num_in_thread, m, n);
                            x_loc--;
                        }
                        __threadfence();
                        // Lock-free based inter-blocks Synchronization
                        __gpu_sync_range_update(block_begin,block_end,step+1,Array_in,Array_out);
                    }
                }
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

void multi_block (Stream& stream_mul, float& total_milliseconds, int limitation, thrust::host_vector<int>& queue_h, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    vector<int> i_0_array (queue_h.size(),0);
    vector<int> i_max_array (queue_h.size(),0);
    vector<int> j_0_array (queue_h.size(),0);
    vector<int> j_max_array (queue_h.size(),0);
    vector<int> blocks_per_table (queue_h.size(),0);
    vector<int> column_major (queue_h.size(),0);


    for (int i=0; i<queue_h.size();i++){
        int row = queue_h[i] / L;
        int column = queue_h[i] % L;
        i_0_array[i] = x_kr[row];
        j_0_array[i] = y_kr[column];
        i_max_array[i] = x_orl[i_0_array[i]] + 1;
        j_max_array[i] = y_orl[j_0_array[i]] + 1;
        if(i_max_array[i]-i_0_array[i]+1 > j_max_array[i]-j_0_array[i]+1){
            blocks_per_table[i] = 2;
            column_major[i] = 1;
        }else{
            blocks_per_table[i] = 2;
            column_major[i] = 0;
        }
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

//    printf("--- Blocks =: %u \n", queue_h.size());

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_mul.cuda_stream());


    multi_block_parallel<<<grid, block, 0, stream_mul.cuda_stream()>>>(limitation, p_d_column_major, p_d_blocks_per_table, queue_h.size(), p_x_orl_d, p_y_orl_d, p_d_i_0_array, p_d_i_max_array, p_d_j_0_array, p_d_j_max_array, p_Delta_d, p_D_d, p_D_tree_d, n, m);


    cudaEventRecord(stop, stream_mul.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
//    printf("Multi-Blocks = %.3fms\n", milliseconds);
    total_milliseconds+=milliseconds;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initial_last(int* D, int n, int m){
    int width = n+1;
    for(int i = TID_1D; i<=n;i+=TOTAL_THREADS_1D){
        D[m*width+n-i] = i;
    }
    for(int i = TID_1D; i <= m; i+= TOTAL_THREADS_1D){
        D[(m-i)*width+n] = i;
    }
}

__device__ void last_table_unit(int* x_orl, int* y_orl, int i, int j, int* Delta, int* D, int* D_tree, int m, int n){
    int width = n+1;
    int index = j + i*width;
    if ((i <= m-1) & (i >= 0) & (j <= n-1) & (j >= 0)){
        if ((x_orl[i] == m-1) & (y_orl[j] == n-1)) {
            D[index] = min3_D(Delta[j+i*n] + D[1 + width + index],
                              1 + D[width+index],
                              1 + D[1 + index]);
            D_tree[j + i*n] = D[index];
        }else{
            D[index] = min3_D(D_tree[j + i*n] + D[y_orl[j] + 1 + (x_orl[i] + 1) * width],
                              1 + D[index+width],
                              1 + D[1 + index]);
        }
    }
}



__global__ void last_table_parallel(int number_per_thread, int* x_orl, int* y_orl, int* Delta, int* D, int* D_tree, int n, int m){

    int y_loc = n - number_per_thread * blockIdx.x;
    int x_begin, x_end;
    int iteration = (m + 1024) / 1024;
    for (int f = 0; f < iteration; f++) {
        x_begin = m- f*1024;
        if (f < iteration - 1) {
            x_end = x_begin - 1024+1;
        } else {
            x_end = 0;
        }
        int x_loc = x_begin-threadIdx.x;
        int k = 0;
        for (int step = 0; step < x_begin + n - x_end + 1; step++) {
            if ((x_loc + y_loc == x_begin + n - step) & (k < units_per_thread) & y_loc >= 0 & x_loc >= x_end) {
                last_table_unit(x_orl, y_orl, x_loc, y_loc, Delta, D, D_tree, m, n);
                y_loc--;
                k++;
            }
            __gpu_sync(step + 1, Array_in, Array_out);
        }
    }
}


void last_table(Stream& stream_sing, float& total_milliseconds, int i, int thread_in_number, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, int L, int blockSize, int n, int m, int *p_x_orl_d, int *p_y_orl_d, int *p_Delta_d, int *p_D_d, int *p_D_tree_d){
    int row = i / L;
    int column = i % L;

    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;

    int one_block = 256;
    int one_grid = 4;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_sing.cuda_stream());
    initial_last<<<one_grid, one_block, 0, stream_sing.cuda_stream()>>>(p_D_d, n, m);
    cudaEventRecord(stop, stream_sing.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    total_milliseconds+=milliseconds;


    int test_block = 1024;
    int test_grid = 64;
    int number_per_thread = (n+test_grid-1)/test_grid;

    cudaMemcpyToSymbol(units_per_thread, &number_per_thread, sizeof(int));


    milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_sing.cuda_stream());

    last_table_parallel<<<test_grid, test_block, 0, stream_sing.cuda_stream()>>>(number_per_thread, p_x_orl_d, p_y_orl_d, p_Delta_d, p_D_d, p_D_tree_d, n, m);

    cudaEventRecord(stop, stream_sing.cuda_stream());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    total_milliseconds+=milliseconds;

}
