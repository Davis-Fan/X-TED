#include "TED_C++.h"

// Two threads for each table (half-half computing)

void initial_1_4(vector<vector<int>>& D, int i_max, int i_0, int j_max){
    int i;
    for (i=i_max-1; i>i_0-1; i--){
        D[i][j_max] = 1 + D[i+1][j_max];
    }
}

void initial_2_4(vector<vector<int>>& D, int j_max, int j_0, int i_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){
        D[i_max][j] = 1 + D[i_max][j+1];
    }
}

void compute_4(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int j, int i_0, int j_0){

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

void task_4(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int i_0, int j_0, int j_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){

//        while ((D[i+1][j+1] == -1)
//                || (D[i+1][j] == -1)
//                || (D[i][j+1] == -1)
//                || (D[x_orl[i]+1][y_orl[j]+1] == -1)){
//            continue;
//        }
        compute_4(Delta, D, D_tree, x_orl, y_orl, i, j, i_0, j_0);
    }
}

void worker1(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int i_max, int j_max){
    int f = i_max - 1;
    int z = i_0;
    if ((f - z) % 2 == 1){
        int stop = (f - z + 1) /2;
        while(f > stop){
            task_4(Delta, D, D_tree, x_orl, y_orl, f, i_0, j_0, j_max);
            f--;
        }
    }else{
        int stop = (f - z) /2;
        while(f > stop){
            task_4(Delta, D, D_tree, x_orl, y_orl, f, i_0, j_0, j_max);
            f--;
        }
    }
}

void worker2(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int i_max, int j_max){
    int f = i_max - 1;
    int z = i_0;
    if ((f - z) % 2 == 1){
        int stop = (f - z + 1) /2;
        while(stop >= z){
            task_4(Delta, D, D_tree, x_orl, y_orl, stop, i_0, j_0, j_max);
            stop--;
        }
    }else{
        int stop = (f - z) /2;
        while(stop >= z){
            task_4(Delta, D, D_tree, x_orl, y_orl, stop, i_0, j_0, j_max);
            stop--;
        }
    }
}


vector<vector<int>> parallel_standard_ted_2_4(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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


            thread t1(initial_1_4, ref(D), i_max, i_0, j_max);
            thread t2(initial_2_4, ref(D), j_max, j_0, i_max);

            t1.join();
            t2.join();

            if ((i_max - 1) != i_0) {
                thread t3(worker1, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);
                thread t4(worker2, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, i_max, j_max);

                t3.join();
                t4.join();
            } else{
                task_4(Delta, D, D_tree, x_orl, y_orl, (i_max-1), i_0, j_0, j_max);
            }

            // Clear D
//            for(auto &a: D){
//                fill(a.begin(),a.end(),0);
//            }
//            if ((k=K-1)&&(l=L-1)){
//                cout << "hello" << endl;
//            }

            for (i=i_max; i>i_0-1; i--){
                for (j=j_max; j>j_0-1; j--){
                    D[i][j] = -1;
                }
            }

        }

    }


    vector<vector<int>> final_result = D_tree;
    return final_result;
}