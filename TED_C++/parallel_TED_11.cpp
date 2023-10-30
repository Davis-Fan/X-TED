#include "TED_C++.h"
//#include <fstream>

// A thread for each table & Interval: 1

//atomic<int> compute_number(0);

void compute_11_1(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max, int x_offset){

    if ((i == i_max) && (j == j_max)){
        D_forest_new[x_offset + i][j] = 0;

    }else if ((i <= i_max-1) && (i >= i_0) && (j == j_max)){
        D_forest_new[x_offset + i][j_max] = 1 + D_forest_new[x_offset + i + 1][j_max];

    }else if ((j <= j_max-1) && (j >= j_0) && (i == i_max)){
        D_forest_new[x_offset + i_max][j] = 1 + D_forest_new[x_offset + i_max][j + 1];

    }else if ((i <= i_max-1) && (i >= i_0) && (j <= j_max-1) && (j >= j_0)){
        if ((x_orl[i] == x_orl[i_0]) && (y_orl[j] == y_orl[j_0])) {

            D_forest_new[x_offset + i][j] = min3(Delta[i][j] + D_forest_new[x_offset + i + 1][j + 1], 1 + D_forest_new[x_offset + i + 1][j], 1 + D_forest_new[x_offset + i][j + 1]);
//            D_tree[i][j] = D_forest_new[x_offset + i][j];
            __atomic_store_n(&(D_tree[i][j]),D_forest_new[x_offset + i][j],__ATOMIC_SEQ_CST);
        } else {
            int val;
            while ((val=__atomic_load_n(&(D_tree[i][j]),__ATOMIC_SEQ_CST)) == -1){
                continue;
            }

            D_forest_new[x_offset + i][j] = min3(val + D_forest_new[x_offset + x_orl[i] + 1][y_orl[j] + 1], 1 + D_forest_new[x_offset + i + 1][j],
                                                 1 + D_forest_new[x_offset + i][j + 1]);
        }
    }

}

void compute_11(int n, int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr,vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree, int L, int m){
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

    int x_offset = (k*L+l) * (m+1);
//    int x_offset = 0;

//    for (int i = m; i>=0; i--){
//        for (int j = n; j>=0; j--){
//            compute_11_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j, i_0, i_max, j_0, j_max, x_offset);
//            compute_number++;
//        }
//    }

    compute_11_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i_max, j_max, i_0, i_max, j_0, j_max, x_offset);
//    compute_number++;


    for (i = i_max - 1; i > i_0 - 1; i--) {
//        D[i][j_max] = 1 + D[i + 1][j_max];
        compute_11_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j_max, i_0, i_max, j_0, j_max, x_offset);
//        compute_number++;
    }

    for (j = j_max - 1; j > j_0 - 1; j--) {
//        D[i_max][j] = 1 + D[i_max][j + 1];
        compute_11_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i_max, j, i_0, i_max, j_0, j_max, x_offset);
//        compute_number++;
    }

    for (i = i_max - 1; i > i_0 - 1; i--) {
        for (j = j_max - 1; j > j_0 - 1; j--) {

            compute_11_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j, i_0, i_max, j_0, j_max, x_offset);
//            compute_number++;
        }
    }


}

void task_11_1(int n, int m, int number, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree){
    number--;
    int row = number / L;
    int column = number % L;
    compute_11(n, row,column,x_orl,x_kr,y_orl,y_kr,Delta, D_forest_new, D_tree, L, m);
}

void task_11(int n, int i_begin, int interval, int final, int m, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree) {
    int i = i_begin;
    while(i<=final){
        task_11_1(n, m,i,L,x_orl,x_kr,y_orl,y_kr,Delta,D_forest_new,D_tree);
        i = i+interval;
    }
}

vector<vector<int>> parallel_standard_ted_2_11(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;
    cout << L*K << endl;

    vector<vector<int>> D_forest_new (K*L*(m+1),vector<int>(n+1,-1));
    vector<thread> threads_pool;

    auto start_time = chrono::steady_clock::now();

    int numbers = num_threads;
    cout << "K*L = "<< K*L << endl;

    for (int inter=1; inter<=numbers; inter++){
        threads_pool.push_back(thread(task_11, n, inter, numbers, K*L, m, L, ref(x_orl),ref(x_kr),ref(y_orl),ref(y_kr),ref(Delta), ref(D_forest_new), ref(D_tree)));
    }

    cout << "Threads' Size = " << threads_pool.size() << endl;

//
    for (thread& th: threads_pool)
    {
        if (th.joinable()) {
            th.join();
        }
    }

    auto end_time = chrono::steady_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Parallel-11 Concurrent task finish, " << ms << " ms consumed" << endl;
    cout << endl;

//    cout << "computer number = " << compute_number << endl;
//    cout << "total number = " << K*L*(m+1)*(n+1) <<endl;
//    cout << "percentage = " << (double) compute_number/(K*L*(m+1)*(n+1)) << endl;


    vector<vector<int>> final_result = D_tree;
    return final_result;
}



