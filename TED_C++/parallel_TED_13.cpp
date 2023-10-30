#include "TED_C++.h"
//#include <fstream>

// A thread for each table & Interval: 1

//atomic<int> hit(0);
//atomic<int> no_need(0);
//int hit = 0;
//int no_need = 0;

void compute_13(vector<int>& level, vector<vector<int>>& Map, int& count, int n, int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

//    count = count + (i_max-i_0+1)*(j_max-j_0+1);
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
                Map[i][j] = n+1;
//                cout << "write" << endl;
//                cout << "Table No. = " << Map[i][j] << endl;
//                cout << "i = " << i << ", j = " << j << endl;
//                cout << endl;
//
            } else {
                int val;
//                auto start = chrono::steady_clock::now();
//                hit++;
//                if ((val=__atomic_load_n(&(D_tree[i][j]),__ATOMIC_SEQ_CST)) == -1){
//                    no_need++;
//                }
//

                while ((val=__atomic_load_n(&(D_tree[i][j]),__ATOMIC_SEQ_CST)) == -1){
                    continue;
                }


//                auto end = chrono::steady_clock::now();
//                auto ms_duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
//                ms_f = ms_f + ms_duration;


                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

//                cout << "read" << endl;
//                if (Map[i][j]>((n+1)/5 * 5)) {
//                cout << "Table No." << n + 1 << " read from Table No." << Map[i][j] << endl;
//                if (level[n+1]<=level[Map[i][j]]){
//                    level[n+1] = level[Map[i][j]]+1;
//                }
//                }
//                cout << "i = " << i << ", j = " << j << endl;
//                cout << endl;

//                if ((D[i][j] == 1 + D[i + 1][j])||(D[i][j] == 1 + D[i][j + 1])){
//                    no_need ++;
//                }

            }

        }
    }


//    for (i=i_max; i>i_0-1; i--){
//        for (j=j_max; j>j_0-1; j--){
//            D[i][j] = -1;
//        }
//    }
}

void task_13_1(vector<int>& level, vector<vector<int>>& Map, int& count, int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
    n--;
    int row = n / L;
    int column = n % L;
    compute_13(level, Map, count, n, row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}


void task_13(vector<vector<int>>& Map, int i_begin, int interval, int final, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree) {
    int i = i_begin;
    int count = 0;
    vector<int> level(final+1, 1);
    level[0] = 0;
    while (i<=final){
        task_13_1(level, Map, count, i,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        i = i+interval;
    }

    vector<int> location;
    location.push_back(0);
//    int s=1;
//    int times = 0;
//    for (int j=1; j<=final; j++){
//        if (level[j] == s) {
//            times++;
//        } else {
//            location.push_back(times);
//            s++;
//            times = 1;
//            if (j==final){
//                location.push_back(times);
//            }
//        }
//    }
    int s = 1;
    int k = level[final];
    cout << endl;
    while(true){
        int time=0;
        for (int j=1;j<=final; j++){
            if (level[j] == s){
                time++;
            }
        }
        location.push_back(time);
        s++;
        if(level[final]<s){
            break;
        }
    }

    cout << endl;
//    cout << "Thread NO."<<i_begin << ", computing times: " << count << endl;
//    printf("Thread NO.%d, computing times: %d \n", i_begin, count);

}



vector<vector<int>> parallel_standard_ted_2_13(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;

//    cout << "Total: " <<K*L << endl;

    vector<vector<int>> Map (m, vector<int>(n,-1));

    vector<thread> threads_pool;
    vector<vector<vector<int>>> D_in_total;
    int numbers = num_threads;

    for (int i=0;i<numbers;i++){
        vector<vector<int>> D_new(m+1, vector<int>(n+1,-1));
        D_in_total.push_back(D_new);
    }
    auto start_time = chrono::steady_clock::now();


    for (int inter=1; inter<=numbers; inter++){
        threads_pool.push_back(thread(task_13, ref(Map), inter, numbers, K*L, L, ref(x_orl),ref(x_kr),ref(y_orl),ref(y_kr),ref(Delta), ref(D_in_total[inter-1]), ref(D_tree)));
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
    cout << "Parallel-13 Concurrent task finish, " << ms/1000.0 << " ms consumed" << endl;
    Total_time = Total_time + ms;
    cout << endl;

//    cout << "hit times = " << hit << endl;
//    cout << "no_need times = " << no_need << endl;
//    cout << "percentage = " << (double)no_need/hit << endl;

    vector<vector<int>> final_result = D_tree;
    return final_result;

}



