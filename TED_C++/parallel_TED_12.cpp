#include "TED_C++.h"

// A thread for each table & Interval: 1

void compute_12(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree, int i, int j, int i_0, int i_max, int j_0, int j_max, int x_offset){
//    int x_offset = (k*L+l) * (m+1);
//    int x_offset = 0;


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

            D_forest_new[x_offset + i][j] = min3(D_tree[i][j] + D_forest_new[x_offset + x_orl[i] + 1][y_orl[j] + 1], 1 + D_forest_new[x_offset + i + 1][j],
                                                 1 + D_forest_new[x_offset + i][j + 1]);
        }
    }

}

void task_12_1(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree, int i, int j, int number, int L, int m){
    number--;
    int row = number / L;
    int column = number % L;
    int i_0 = x_kr[row];
    int j_0 = y_kr[column];
    int i_max = x_orl[i_0] + 1;
    int j_max = y_orl[j_0] + 1;

    int x_offset = (row*L+column) * (m+1);
    compute_12(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j, i_0, i_max, j_0, j_max, x_offset);
}


void task_12(int i_begin, int interval, int final, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D_forest_new, vector<vector<int>>& D_tree,int m, int n, int L){
    int k;
    int l;
    int i_0;
    int j_0;
    int i_max;
    int j_max;

// test 1
//    for (int i=m; i>=0; i--){
//        for (int j=n; j>=0; j--){
//            int d = i_begin;
//            while (d<=final){
//                task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j, d, L, m);
//                d = d + interval;
//            }
//        }
//    }

// test 2
//    int d = i_begin;
//    while (d<=final){
//
//        for (int i=m; i>=0; i--){
//            for (int j=n; j>=0; j--){
//                task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, i, j, d, L, m);
//            }
//        }
//        d = d + interval;
//    }


// test 3
    int inv = 0;
    while (n-inv >= 0){

        int i = i_begin;
        while (i<=final){
            task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, m, n-inv, i, L, m);
            i = i+interval;
        }


        int jec = 1;
        while (jec<=inv){

            i = i_begin;
            while (i<=final){
                task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, m-jec, n-inv+jec, i, L, m);
                i = i+interval;
            }

            jec++;
        }
        inv++;
    }


    int inv_2 = 1;
    while (m - inv_2 >= 0){

        int i = i_begin;
        while (i<=final){
            task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, m-inv_2, 0, i, L, m);
            i = i+interval;
        }

        int jec = 1;
        while (m - inv_2 - jec >= 0){

            i = i_begin;
            while (i<=final){
                task_12_1(x_orl, x_kr, y_orl, y_kr, Delta, D_forest_new, D_tree, m-inv_2-jec, 0+jec, i, L, m);
                i = i+interval;
            }

            jec++;
        }
        inv_2++;
    }
}


vector<vector<int>> parallel_standard_ted_2_12(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads){
    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    cout << "Total: " <<K*L << endl;


    vector<vector<int>> D_forest_new (K*L*(m+1),vector<int>(n+1,-1));
    vector<thread> threads_pool;

    auto start_time = chrono::steady_clock::now();

    int numbers = num_threads;

    for (int inter=1; inter<=numbers; inter++){
        threads_pool.push_back(thread(task_12, inter, numbers, K*L, ref(x_orl),ref(x_kr),ref(y_orl),ref(y_kr),ref(Delta), ref(D_forest_new), ref(D_tree), m, n, L));
    }

//    cout << "Threads' Size = " << threads_pool.size() << endl;


    for (thread& th: threads_pool)
    {
        if (th.joinable()) {
            th.join();
        }
    }


    auto end_time = chrono::steady_clock::now();
    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    cout << "Parallel-12 Concurrent task finish, " << ms/1000.0 << " ms consumed" << endl;
    Total_time = Total_time + ms;
    cout << endl;

    vector<vector<int>> final_result = D_tree;

    return final_result;
}


