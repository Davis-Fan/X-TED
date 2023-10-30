#include "TED_C++.h"

// Three threads for each table: row, column and diagonal

void initial_1_2(vector<vector<int>>& D, int& i_max, int& i_0, int& j_max){
    int i;
    for (i=i_max-1; i>i_0-1; i--){
        D[i][j_max] = 1 + D[i+1][j_max];
    }
}

void initial_2_2(vector<vector<int>>& D, int& j_max, int& j_0, int& i_max){
    int j;
    for (j=j_max-1; j>j_0-1; j--){
        D[i_max][j] = 1 + D[i_max][j+1];
    }
}

void compute_2(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i, int j, int i_0, int j_0){

//    while (((D[i+1][j+1] == 0) && (i+1 != i_max) && (j+1 != j_max))
//           || ((D[i+1][j] == 0) && (i+1 != i_max) && (j != j_max))
//           || ((D[i][j+1] == 0) && (i != i_max) && (j+1 != j_max))
//           || ((D[x_orl[i]+1][y_orl[j]+1] == 0) && (x_orl[i]+1 != i_max) && (y_orl[j]+1 != j_max))) {
//        continue;
//    }

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

void diagonal(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int m, int n){
    if((m - i_0 + 1) >= (n - j_0 + 1)){
        int k = 0;
        while (n - k >= j_0){
            compute_2(Delta, D, D_tree, x_orl, y_orl,m-k,n-k, i_0, j_0);
            k++;
        }
    }else{
        int k = 0;
        while (m - k >= i_0){
            compute_2(Delta, D, D_tree, x_orl, y_orl, m-k, n-k, i_0, j_0);
            k++;
        }
    }
}

void row(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int m, int n){
    if((m - i_0 + 1) >= (n - j_0 + 1)){

        int k = 0;
        while ((n - k - 1) >= j_0) {
            int f = 0;
            while (n - k - 1 - f >= j_0){
                compute_2(Delta, D, D_tree, x_orl, y_orl,m-k,n-k-1-f, i_0, j_0);
                f++;
            }
           k++;
        }

    }else{

        int k = 0;
        while (m - k >= i_0) {
            int f = 0;
            while (n - k - 1 - f >= j_0){
                compute_2(Delta, D, D_tree, x_orl, y_orl, m-k, n-k-1-f, i_0, j_0);
                f++;
            }
            k++;

        }



    }
}

void column(vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, vector<int>& x_orl, vector<int>& y_orl, int i_0, int j_0, int m, int n){
    if((m - i_0 + 1) >= (n - j_0 + 1)){

        int k = 0;
        while (n - k >= j_0){
            int f = 0;
            while (m - k - 1 - f >= i_0){
                compute_2(Delta,D,D_tree,x_orl,y_orl,m-k-1-f,n-k,i_0,j_0);
                f++;
            }
            k++;
        }

    }else{

        int k = 0;
        while (m - k - 1 >= i_0){
            int f = 0;
            while (m - k - 1 - f >= i_0){
                compute_2(Delta,D,D_tree,x_orl,y_orl,m-k-1-f,n-k,i_0,j_0);
                f++;
            }
            k++;
        }


    }
}


vector<vector<int>> parallel_standard_ted_2_2(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

                thread t1(initial_1_2, ref(D), ref(i_max), ref(i_0), ref(j_max));
                thread t2(initial_2_2, ref(D), ref(j_max), ref(j_0), ref(i_max));

                t1.join();
                t2.join();


                int x = i_max - 1;
                int y = j_max - 1;
                thread t3(diagonal, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                thread t4(row, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                thread t5(column, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                t3.join();
                t4.join();
                t5.join();

                auto end_time = chrono::steady_clock::now();
                auto ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
                cout << "Last Table-2, " << ms << " ms consumed" << endl;

            }else {

                thread t1(initial_1_2, ref(D), ref(i_max), ref(i_0), ref(j_max));
                thread t2(initial_2_2, ref(D), ref(j_max), ref(j_0), ref(i_max));

                t1.join();
                t2.join();


                int x = i_max - 1;
                int y = j_max - 1;
                thread t3(diagonal, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                thread t4(row, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                thread t5(column, ref(Delta), ref(D), ref(D_tree), ref(x_orl), ref(y_orl), i_0, j_0, x, y);
                t3.join();
                t4.join();
                t5.join();

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
