#include "TED_C++.h"

// Two threads for each table (interleaved computing)

void initial_1_3(vector<vector<int>>& D, int i_max, int i_0, int j_max){
    int i;
    for (i=i_max-1; i>i_0-1; i--){
        D[i][j_max] = 1 + D[i+1][j_max];
    }
}

void initial_2_3(vector<vector<int>>& D, int j_max, int j_0, int i_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){
        D[i_max][j] = 1 + D[i_max][j+1];
    }
}


void compute_3(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int j, int i_0, int j_0){

    if ((x_orl[i] == x_orl[i_0]) & (y_orl[j] == y_orl[j_0])){

        while ((D[i+1][j+1] == -1)
               || (D[i+1][j] == -1)
               || (D[i][j+1] == -1)){
            continue;
        }

        D[i][j] = min3(Delta[i][j] + D[i+1][j+1], 1 + D[i+1][j], 1 + D[i][j+1]);
        D_tree[i][j] = D[i][j];

    }else{

        while ((D[i+1][j] == -1)
               || (D[i][j+1] == -1)
               || (D[x_orl[i]+1][y_orl[j]+1] == -1)){
            continue;
        }

        D[i][j] = min3(D_tree[i][j] + D[x_orl[i]+1][y_orl[j]+1], 1 + D[i+1][j], 1 + D[i][j+1]);

    }

}

void task_3(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int i_0, int j_0, int j_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){

//        while ((D[i+1][j+1] == -1)
//                || (D[i+1][j] == -1)
//                || (D[i][j+1] == -1)
//                || (D[x_orl[i]+1][y_orl[j]+1] == -1)){
//            continue;
//        }
        compute_3(Delta, D, D_tree, x_orl, y_orl, i, j, i_0, j_0);
    }
}

void row1(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int i_max, int j_max){
    int f = i_max - 1;
    int z = i_0;
    if ((f - z) % 2 == 1){
        int i = f;
        while (i >= z + 1){
            task_3(Delta, D, D_tree, x_orl, y_orl, i, i_0, j_0, j_max);
            i = i - 2;
        }
    }else{
        int i = f;
        while (i >= z){
            task_3(Delta, D, D_tree, x_orl, y_orl, i, i_0, j_0, j_max);
            i = i - 2;
        }
    }
}

void row2(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int i_max, int j_max){
    int f = i_max - 1;
    int z = i_0;
    if ((f - z) % 2 == 1){
        int i = f;
        while ((i-1) >= z){
            task_3(Delta, D, D_tree, x_orl, y_orl, i-1, i_0, j_0, j_max);
            i = i - 2;
        }
    }else{
        int i = f;
        while ((i-1) >= (z+1)){
            task_3(Delta, D, D_tree, x_orl, y_orl, i-1, i_0, j_0, j_max);
            i = i - 2;
        }
    }

}

vector<vector<int>> parallel_standard_ted_2_3(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    int m = (int)x_orl.size();
//    int n = (int)y_orl.size();

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;

    int i;
    int j;

    int i_0;
    int j_0;
    int i_max;
    int j_max;

    for (k=0; k<K; k++){
        for (l=0; l<L; l++){

            i_0 = x_kr[k];
            j_0 = y_kr[l];
            i_max = x_orl[i_0] + 1;
            j_max = y_orl[j_0] + 1;
            D[i_max][j_max] = 0;

            if ((k == K-1)&&(l == L-1)){


                auto start_time = chrono::steady_clock::now();

                thread t1(initial_1_3, ref(D), i_max, i_0, j_max);
                thread t2(initial_2_3, ref(D), j_max, j_0, i_max);
                t1.join();
                t2.join();


                if ((i_max - 1) != i_0) {

                    thread t3(row1, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);
                    thread t4(row2, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);

                    t3.join();
                    t4.join();

                } else {
                    task_3(Delta, D, D_tree, x_orl, y_orl, (i_max - 1), i_0, j_0, j_max);
                }

                auto end_time = chrono::steady_clock::now();
                auto ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
                cout << "Last Table-3, " << ms << " ms consumed" << endl;

            }else {

                thread t1(initial_1_3, ref(D), i_max, i_0, j_max);
                thread t2(initial_2_3, ref(D), j_max, j_0, i_max);
                t1.join();
                t2.join();


                if ((i_max - 1) != i_0) {

                    thread t3(row1, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);
                    thread t4(row2, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);

                    t3.join();
                    t4.join();

                } else {
                    task_3(Delta, D, D_tree, x_orl, y_orl, (i_max - 1), i_0, j_0, j_max);
                }



                // Clear D
//            for(auto &a: D){
//                fill(a.begin(),a.end(),0);
//            }
//            if ((k=K-1)&&(l=L-1)){
//                cout << "hello" << endl;
//            }
            }


            for (i=i_max; i>i_0-1; i--){
                for (j=j_max; j>j_0-1; j--){
                    D[i][j] = -1;
                }
            }
//            thread t5(initial_3, ref(D), i_0, j_0, i_max, j_max);
//            t5.join();

        }

    }


    vector<vector<int>> final_result = D_tree;
    return final_result;
}