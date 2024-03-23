#include "TED_C++.h"

// Compute each table
void compute(int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Cost, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    int i;
    int j;

    int i_0;
    int j_0;
    int i_max;
    int j_max;

    i_0 = x_kr[k];
    j_0 = y_kr[l];
    i_max = x_orl[i_0] + 1;
    j_max = y_orl[j_0] + 1;
    D[i_max][j_max] = 0;


    for (i = i_max - 1; i > i_0 - 1; i--) {
        D[i][j_max] = 1 + D[i + 1][j_max];
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
        D[i_max][j] = 1 + D[i_max][j + 1];
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {

            if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])) {

                D[i][j] = min3(Cost[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
                D_tree[i][j] = D[i][j];

            } else {

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

            }
        }
    }

}

// Fetch task
void compute_one_table( int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Cost, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    int row = n / L;
    int column = n % L;
    compute(row,column,x_orl,x_kr,y_orl,y_kr,Cost, D, D_tree);
}


// Parallel computing
void task(vector<int>& depth,  vector<int>& worklist_1, vector<int>& worklist_2, int begin, int interval, int final, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Cost, vector<vector<int>>& D, vector<vector<int>>& D_tree) {
    int i = begin;
    while (i < final) {
        int task = worklist_1[i];
        worklist_1[i] = -1;
        compute_one_table(task, L, x_orl, x_kr, y_orl, y_kr, Cost, D, D_tree);
        i = i + interval;
    }
}

vector<vector<int>> parallel_cpu_ted(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Cost, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int num_th = num_threads;
    vector<vector<vector<int>>> D_in_total;
    for (int i=0;i<num_th;i++){
        vector<vector<int>> D_new(m+1, vector<int>(n+1,-1));
        D_in_total.push_back(D_new);
    }

    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();

    vector<int> x_keyroot_depth(number_x_nodes, -1);
    vector<int> y_keyroot_depth(number_y_nodes, -1);

    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

//    auto start_time_2 = chrono::steady_clock::now();

    // Preprocessing begins

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

    for (int i=0;i<K*L;i++){
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
    }

    // Preprocessing ends

//    auto end_time_2 = chrono::steady_clock::now();
//    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
//    cout << "Preprocessing Method 18, the time is " << ms_2/1000.0 << " ms consumed" << endl;


    vector<int> worklist1(K*L,-1);
    vector<int> worklist2(K*L,-1);
    int worklist1_tail=0;
    int worklist2_tail=0;

    int current_depth = 0;
    for (int i=0; i<K*L; i++){
        if (depth[i] == current_depth){
            worklist1[worklist1_tail++] = i;
        }
    }

    int max = 0;
    for (int i=0; i < depth.size(); i++){
        if (max < depth[i]){
            max = depth[i];
        }
    }


    double total_time = 0;


    while (worklist1_tail != 0) {
        auto start_time = chrono::steady_clock::now();

        vector<thread> threads;

        for (int inter = 0; inter < num_th; inter++) {
            threads.push_back(thread(task, ref(depth), ref(worklist1), ref(worklist2), inter, num_th, (int) worklist1_tail, L, ref(x_orl),ref(x_kr), ref(y_orl), ref(y_kr), ref(Cost), ref(D_in_total[inter]), ref(D_tree)));
        }

        for (auto &th: threads) {
            th.join();
        }

        auto end_time = chrono::steady_clock::now();
        auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
        total_time += static_cast<double>(ms/1000.0);

        current_depth++;
        for (int i=0; i<K*L; i++){
            if (depth[i] == current_depth){
                worklist2[worklist2_tail++] = i;
            }
        }

        swap(worklist1, worklist2);
        worklist1_tail = worklist2_tail;
        worklist2_tail = 0;
    }

    cout << "Total Time for Parallel Computing = " << total_time <<" ms"<< endl;
    vector<vector<int>> final_result = D_tree;
    return final_result;
}