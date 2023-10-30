#include "TED_C++.h"
// For optimizing preprocessing

void compute_15(int n, int k, int l, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
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
                D_tree[i][j] = D[i][j];

            } else {

                D[i][j] = min3(D_tree[i][j] + D[x_orl[i] + 1][y_orl[j] + 1], 1 + D[i + 1][j],
                               1 + D[i][j + 1]);

            }
        }
    }

}

void task_15_1(int n, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree){
//    n--;
    int row = n / L;
    int column = n % L;
    compute_15(n, row,column,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
}

void task_15(vector<bool>& compute, vector<int>& depth, vector<int>& index, vector<int>& adjacent_list,  vector<int>& worklist_1, vector<int>& worklist_2, atomic_int& worklist2_tail, int i_begin, int interval, int final, int L, vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree) {
    int i = i_begin;
    while (i < final) {
        int task = worklist_1[i];
        compute[task] = 1;
        worklist_1[i] = -1;
        task_15_1(task,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);
        for (int j = index[task]; j < index[task + 1]; j++) {
            if (depth[task] == depth[adjacent_list[j]] - 1 & compute[adjacent_list[j]] == 0) {
                worklist_2[worklist2_tail++] = adjacent_list[j];
//                worklist_2.push_back(adjacent_list[j]);
                compute[adjacent_list[j]] = 1;
            }
        }
        i = i + interval;
    }
}

vector<string> split(const string& str, const string& delim) {
    vector<string> res;
    if("" == str) return res;
    char * strs = new char[str.length() + 1] ;
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p) {
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }

    return res;
}


vector<vector<int>> parallel_standard_ted_2_15(vector<int>& x_orl, vector<int>& x_kr, vector<int>& y_orl, vector<int>& y_kr, vector<vector<int>>& Delta, vector<vector<int>>& D, vector<vector<int>>& D_tree, int m, int n, int num_threads){

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

    vector<std::pair<int,int>> Table_index;
    for (int i=0; i<K; i++){
        for (int j=0; j<L; j++){
            Table_index.emplace_back(x_kr[i], y_kr[j]);
        }
    }


    int num_depended_table = 0;
    vector<int> index(K*L+1, 0);
    vector<int> adjacent_list;
    vector<int> depth(K*L, 1);
    vector<bool> compute(K*L, 0);

    vector<bool> x_leaf_nodes(K*L, false);
    vector<bool> y_leaf_nodes(K*L, false);

    // Find leaf node
    for (int i=K-1;i>=0; i--){
        int node = x_kr[i];
        if(node == x_orl[node]){
            x_leaf_nodes[node] = true;
        }
    }

    for (int i=L-1;i>=0; i--){
        int node = y_kr[i];
        if(node == y_orl[node]){
            y_leaf_nodes[node] = true;
        }
    }

    // select related tables
    set<int> table_No;
    for (int i=0; i<K*L;i++){
        if (!x_leaf_nodes[Table_index[i].first] || !y_leaf_nodes[(Table_index[i].second)]) {
            table_No.insert(i);
        }else{
            depth[i] = 0;
        }
    }


    size_t size_1=0,size_2=0;

//    auto start_time = chrono::steady_clock::now();
//
//    for (auto k:table_No){
//        auto it = table_No.lower_bound(k);
//        for (auto rit = set<int>::reverse_iterator(it); rit != table_No.rend(); ++rit) {
//            int j = *rit;
//            if (Table_index[j].first >= Table_index[k].first & Table_index[j].first <= x_orl[Table_index[k].first] & Table_index[j].second >= Table_index[k].second & Table_index[j].second <= y_orl[Table_index[k].second]) {
//                depth[k] = depth[j] + 1;
////                break;
//            }
//            size_1++;
//        }
//    }
//    auto end_time = chrono::steady_clock::now();
//    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
//    cout << "Method 1, " << ms/1000.0 << " ms consumed" << endl;

    vector<bool> is_leaf_Table(K*L, false);
    for (int i=0; i<K*L;i++){
        if (x_leaf_nodes[Table_index[i].first] & y_leaf_nodes[Table_index[i].second]){
            is_leaf_Table[i] = true;
        }
    }

    auto start_time_2 = chrono::steady_clock::now();

    #pragma omp parallel for
    for (int i=1; i<K*L; i++){
        if(is_leaf_Table[i]){
            continue;
        } else{
            for (int j=i-1; j>=0; j--){
                if (is_leaf_Table[j]){
                    continue;
                }else {
                    if (Table_index[j].first >= Table_index[i].first &
                        Table_index[j].first <= x_orl[Table_index[i].first] &
                        Table_index[j].second >= Table_index[i].second &
                        Table_index[j].second <= y_orl[Table_index[i].second]) {
                        depth[i] = max(depth[j] + 1, depth[i]);
                    }
//                    size_2++;
                }
            }
        }
    }
    double dayi = (double)K*L*(K*L-1)/2;
    cout << "ratio = " << size_2/dayi << endl;

    auto end_time_2 = chrono::steady_clock::now();
    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
    cout << "Method 2, the time is " << ms_2/1000.0 << " ms consumed" << endl;

    cout << "Size 1 = " << size_1 << endl;
    cout << "Size 2 = " << size_2 << endl;





//    int x,y;
//    vector<int> depth_2(K*L, 0);
//    for (int i=0; i<K*L; i++){
//        x = Table_index[i].first;
//        y = Table_index[i].second;
//        for (int j=i+1; j<K*L; j++){
//            x_subtree_left = Table_index[j].first;
//            x_subtree_right = x_orl[x_subtree_left];
//            y_subtree_left = Table_index[j].second;
//            y_subtree_right = y_orl[y_subtree_left];
//
//            if (x >=x_subtree_left & x <=x_subtree_right & y >= y_subtree_left & y <= y_subtree_right){
//                    adjacent_list.push_back(j);
//                    num_depended_table++;
//                    if (depth_2[j] <= depth_2[i] + 1){
//                        depth_2[j] = depth_2[i] + 1;
//                    }
//                }
//            }
//        index[i+1] = num_depended_table;
//    }

//    ofstream f("/Users/davis/CLionProjects/TED_C++/depth.txt", ios::app);
//    for (int j = 0; j <depth.size(); j++) {
//        f<<depth[j]<<" ";
//    }
//    f.close();
//
//    ofstream f2("/Users/davis/CLionProjects/TED_C++/compute.txt", ios::app);
//    for (int j = 0; j <compute.size(); j++) {
//        f2<<compute[j]<<" ";
//    }
//    f2.close();
//
//    ofstream f3("/Users/davis/CLionProjects/TED_C++/index.txt", ios::app);
//    for (int j = 0; j <index.size(); j++) {
//        f3<<index[j]<<" ";
//    }
//    f3.close();
//
//    ofstream f4("/Users/davis/CLionProjects/TED_C++/adjacent_list.txt", ios::app);
//    for (int j = 0; j <adjacent_list.size(); j++) {
//        f4<<adjacent_list[j]<<" ";
//    }
//    f4.close();

//    vector<int> depth;
//    ifstream f("/Users/davis/CLionProjects/TED_C++/depth.txt");
//    string buff;
//    getline(f,buff);
//    std::vector<string> res = split(buff, " ");
//    for(int i=0; i<res.size(); i++){
//        int k = std::stoi(res[i]);
//        depth.push_back(k);
//    }
//    f.close();
//
//    vector<bool> compute;
//    ifstream f2("/Users/davis/CLionProjects/TED_C++/compute.txt");
//    string buff2;
//    getline(f2,buff2);
//    std::vector<string> res2 = split(buff2, " ");
//    for(int i=0; i<res2.size(); i++){
//        int k = std::stoi(res2[i]);
//        compute.push_back((bool)k);
//    }
//    f2.close();
//
//    vector<int> index;
//    ifstream f3("/Users/davis/CLionProjects/TED_C++/index.txt");
//    string buff3;
//    getline(f3,buff3);
//    std::vector<string> res3 = split(buff3, " ");
//    for(int i=0; i<res3.size(); i++){
//        int k = std::stoi(res3[i]);
//        index.push_back(k);
//    }
//    f3.close();
//
//    vector<int> adjacent_list;
//    ifstream f4("/Users/davis/CLionProjects/TED_C++/adjacent_list.txt");
//    string buff4;
//    getline(f4,buff4);
//    std::vector<string> res4 = split(buff4, " ");
//    for(int i=0; i<res4.size(); i++){
//        int k = std::stoi(res4[i]);
//        adjacent_list.push_back(k);
//    }
//    f4.close();


    vector<int> worklist1(1000000,-1);
    vector<int> worklist2(1000000,-1);
    atomic_int worklist1_tail;
    atomic_int worklist2_tail;
    worklist1_tail.store(0);
    worklist2_tail.store(0);

    int current_depth = 0;
    for (int i=0; i<K*L; i++){
        if (depth[i] == current_depth){
            worklist1[worklist1_tail++] = i;
        }
    }

//    auto start_time = chrono::steady_clock::now();
//    auto end_time = chrono::steady_clock::now();
//    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
//    cout << "Parallel-15 Concurrent task finish, " << ms/1000.0 << " ms consumed" << endl;
//    Total_time = Total_time + ms;
//    cout << endl;

    cout << "Threads size = " << num_th << endl;
//    auto start_time_2 = chrono::steady_clock::now();
    while (worklist1[0] != -1 && worklist1[0] != K*L-1) {

        if (worklist1_tail >= num_th) {

            vector<thread> threads;

            for (int inter = 0; inter < num_th; inter++) {
                threads.push_back(thread(task_15, ref(compute), ref(depth), ref(index), ref(adjacent_list), ref(worklist1), ref(worklist2), ref(worklist2_tail), inter, num_th, (int) worklist1_tail, L, ref(x_orl),ref(x_kr), ref(y_orl), ref(y_kr), ref(Delta), ref(D_in_total[inter]), ref(D_tree)));
            }

            for (auto &th: threads) {
                th.join();
            }

        } else {

            for (int i = 0; i < worklist1_tail; i++) {
                int task = worklist1[i];
                compute[task] = 1;
                worklist1[i] = -1;
                task_15_1(task, L, x_orl, x_kr, y_orl, y_kr, Delta, D, D_tree);
                for (int j = index[task]; j < index[task + 1]; j++) {
                    if (depth[task] == depth[adjacent_list[j]] - 1 & compute[adjacent_list[j]] == 0) {
                        worklist2[worklist2_tail++] = adjacent_list[j];
                        compute[adjacent_list[j]] = 1;
                    }
                }
            }
        }
        swap(worklist1, worklist2);
        worklist1_tail.store(worklist2_tail);
        worklist2_tail.store(0);
    }
    task_15_1(K*L-1,L,x_orl,x_kr,y_orl,y_kr,Delta, D, D_tree);

//    auto end_time_2 = chrono::steady_clock::now();
//    auto ms_2 = chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count();
//    cout << "Parallel-15 Concurrent task finish, " << ms_2/1000.0 << " ms consumed" << endl;

    vector<vector<int>> final_result = D_tree;
    return final_result;

}





