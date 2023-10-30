#include "TED_C++.h"
//#include <fstream>

// A thread for each table (4 threads interleaved) & Interval: 10
long long ms_f;

void compute_6(int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

                D[i][j] = min3(Delta[i][j] + D[i + 1][j + 1], 1 + D[i + 1][j], 1 + D[i][j + 1]);
//                D_tree[i][j] = D[i][j];
                __atomic_store_n(&(D_tree[i][j]),D[i][j],__ATOMIC_SEQ_CST);

            } else {
                int val;
//                auto start = chrono::steady_clock::now();

                while ((val=__atomic_load_n(&(D_tree[i][j]),__ATOMIC_SEQ_CST)) == -1){
                    continue;
                }


//                auto end = chrono::steady_clock::now();
//                auto ms_duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//                ms_f = ms_f + ms_duration;

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

            }

        }
    }


    for (i=i_max; i>i_0-1; i--){
        for (j=j_max; j>j_0-1; j--){
            D[i][j] = -1;
        }
    }
}

void task_6(int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    n--;
    int row = n / L;
    int column = n % L;
    compute_6(row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void worker_6_1(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int i = 1;
    while(i<=K*L){
        for (int j=0; j<10; j++){
            task_6(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
    i = i + 40;
    }
}

void worker_6_2(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int i = 11;
    while(i<=K*L){
        for (int j=0; j<10; j++){
            task_6(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
        i = i + 40;
        }
    }


void worker_6_3(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    int i = 21;
    while(i<=K*L){
        for (int j=0; j<10; j++){
            task_6(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
        i = i + 40;
    }
}


void worker_6_4(int K, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    int i = 31;
    while(i<=K*L){
        for (int j=0; j<10; j++){
            task_6(i+j, L, x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        }
        i = i + 40;
    }
}


vector<vector<int>> parallel_standard_ted_2_6(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;
//    cout << L*K << endl;

    vector<vector<int>> D_2 = D;
    vector<vector<int>> D_3 = D;
    vector<vector<int>> D_4 = D;

    auto start_time = chrono::steady_clock::now();

    thread t1(worker_6_1, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D), ref(D_tree));
    thread t2(worker_6_2, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_2), ref(D_tree));
    thread t3(worker_6_3, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_3), ref(D_tree));
    thread t4(worker_6_4, K, L, ref(x_orl), ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_4), ref(D_tree));

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    auto end_time = chrono::steady_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Parallel-6 Concurrent task finish, " << ms << " ms consumed" << endl;
    cout << endl;

    cout << "Iteration Time = " << ms_f << " ms consumed" << endl;
    cout << endl;

    vector<vector<int>> final_result = D_tree;
    return final_result;
}

