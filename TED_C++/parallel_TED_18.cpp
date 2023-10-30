#include "TED_C++.h"


void compute_18(vector<vector<vector<int>>>& record, fstream& output_file, double& count, double& max, double& min, int n, int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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

//    printf("\n");
//    printf("table (%d, %d) --- \n", k, l);
//    printf("keyroot_x node = %d, x_max node = %d\n", i_0, i_max);
//    printf("keyroot_y node = %d, y_max node = %d\n", j_0, j_max);


    count = count + (i_max-i_0+1)*(j_max-j_0+1);
    if((i_max-i_0+1)*(j_max-j_0+1) > max){
        max = (i_max-i_0+1)*(j_max-j_0+1);
    }
    if((i_max-i_0+1)*(j_max-j_0+1) < min){
        min = (i_max-i_0+1)*(j_max-j_0+1);
    }

    output_file << (i_max-i_0+1)*(j_max-j_0+1) << endl;
//    printf("%u \n", (i_max-i_0+1)*(j_max-j_0+1));
//
//    if(i_max-i_0+1 > j_max-j_0+1){
//        output_file << (j_max-j_0+1) << endl;
//    }else{
//        output_file << (i_max-i_0+1) << endl;
//    }

//    if((i_max-i_0+1)*(j_max-j_0+1)>3000){
//        if(i_max-i_0+1 > j_max-j_0+1){
//            output_file << (j_max-j_0+1) << endl;
//        }else{
//            output_file << (i_max-i_0+1) << endl;
//        }
//    }else{
//        output_file << 0 << endl;
//    }

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
                D_tree[i][j] = D[i][j];
//                record[0][i][j] = k;
//                record[1][i][j] = l;
//                record[2][i][j] = i;
//                record[3][i][j] = j;
//                printf("table (%d, %d): write D_tree[%d][%d] \n", k, l, i, j);

            } else {

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

//                printf("table(%d,%d) [%d][%d] depends on  Table(%d,%d) [%d][%d] \n", k, l, i, j, record[0][i][j], record[1][i][j], record[2][i][j], record[3][i][j]);
            }
        }
    }

}

void task_18_1( vector<vector<vector<int>>>& record, fstream& output_file, double& count, double& max, double& min, int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    n--;
    int row = n / L;
    int column = n % L;
    compute_18(record, output_file, count, max, min, n, row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void task_18(vector<vector<vector<int>>>& record, fstream& output_file, vector<int>& depth,  vector<int>& worklist_1, vector<int>& worklist_2, int i_begin, int interval, int final, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree) {
    int i = i_begin;
    double count = 0;
    double max = 0;
    double min = 2000000;
//    vector<vector<int>> test(4, vector<int>(4,-1));
//    if (final == 50) {
    while (i < final) {
        int task = worklist_1[i];
        worklist_1[i] = -1;
        task_18_1(record,output_file, count,max, min, task, L, x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree);
        i = i + interval;
    }
}

vector<vector<int>> parallel_standard_ted_2_18(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads, vector<vector<int>>& x_adj, vector<vector<int>>& y_adj){

    int K = (int)x_kr.size();
    int L = (int)y_kr.size();

    int k;
    int l;
    int x_subtree_left;
    int x_subtree_right;
    int y_subtree_left;
    int y_subtree_right;

    int num_th = num_threads;
    vector<vector<vector<int>>> D_in_total;
    for (int i=0;i<num_th;i++){
        vector<vector<int>> D_new(m+1, vector<int>(n+1,-1));
        D_in_total.push_back(D_new);
    }
    double sizeInBytes = sizeof(D_in_total[0][0][0]) * (m+1) * (n+1);
    double sizeInMB = sizeInBytes / (1024 * 1024);

//    std::cout << "Size of vector: " << std::fixed << std::setprecision(3) << sizeInMB << " MB\n";

    vector<int> depth(K*L, 0);

    int number_x_nodes = x_adj.size();
    int number_y_nodes = y_adj.size();

    vector<int> x_keyroot_depth(number_x_nodes, -1);
    vector<int> y_keyroot_depth(number_y_nodes, -1);

    vector<int> x_keyroot_depth_2(K, 0);
    vector<int> y_keyroot_depth_2(L, 0);

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

    for (int i=0;i<K*L;i++){
        depth[i] = x_keyroot_depth_2[i/L] + y_keyroot_depth_2[i%L];
    }

    auto end_time_2 = chrono::steady_clock::now();
    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
    cout << "Preprocessing Method 18, the time is " << ms_2/1000.0 << " ms consumed" << endl;


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

    vector<int> level (max+1,0);
    for (int i =0;i<=max; i++){
        for (int k=0; k<depth.size(); k++){
            if (depth[k] == i){
                level[i]++;
            }
        }
    }

//    cout << "Threads size = " << num_th << endl;
    int total_time = 0;
    fstream output_file("/Users/davis/CLionProjects/TED_C++/Figure14.txt");
//    cout << "depth = " << current_depth << " the number is " << worklist1_tail<< endl;
    output_file << "depth = " << current_depth << " the number is " << worklist1_tail<< endl;
    vector<vector<vector<int>>> record (4,vector<vector<int>>(4, vector<int>(4,-1)));

//    auto start_time = chrono::steady_clock::now();
    while (worklist1_tail != 0) {
        auto start_time = chrono::steady_clock::now();
        if (worklist1_tail >= num_th) {

            vector<thread> threads;

            for (int inter = 0; inter < num_th; inter++) {
                threads.push_back(thread(task_18, ref(record), ref(output_file), ref(depth), ref(worklist1), ref(worklist2), inter, num_th, (int) worklist1_tail, L, ref(x_orl),ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_in_total[inter]), ref(D_tree)));
            }

            for (auto &th: threads) {
                th.join();
            }

        } else {
            for (int i = 0; i < worklist1_tail; i++) {
                int task = worklist1[i];
                worklist1[i] = -1;
                double dayi = 0;
                double dayi2 = 0;
                task_18_1(record, output_file, dayi, dayi2, dayi, task,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
            }
        }
        auto end_time = chrono::steady_clock::now();
        auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
        total_time += static_cast<double>(ms/1000.0);
//        cout <<  static_cast<double>(ms/1000.0) << "ms ";
        current_depth++;
        for (int i=0; i<K*L; i++){
            if (depth[i] == current_depth){
                worklist2[worklist2_tail++] = i;
            }
        }
//        cout << "depth = " << current_depth << " the number is " << worklist2_tail<< endl;
        output_file << "depth = " << current_depth << " the number is " << worklist2_tail<< endl;
        swap(worklist1, worklist2);
        worklist1_tail = worklist2_tail;
        worklist2_tail = 0;
//        auto end_time = chrono::steady_clock::now();
//        auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
//        total_time += static_cast<int>(ms/1000.0);
//        cout <<  static_cast<int>(ms/1000.0) << " ";
//        cout << "Depth " << current_depth<<" of Tables finished, " << ms/1000.0 << " ms consumed" << endl;
    }
    cout << endl;
    cout << "Total Time = " << total_time << endl;
    output_file.close();

//    auto end_time = chrono::steady_clock::now();
//    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
//    cout << "Parallel-18 Concurrent task finish, " << ms/1000.0 << " ms consumed" << endl;

//    printf("%d hello\n", D_tree[0][13]);
//    printf("%d world\n", D_tree[61][0]);

    vector<vector<int>> final_result = D_tree;
    return final_result;

}