#include "TED_C++.h"
void test_3(int num_of_nodes, int num_threads, int parallel_version){
    int num_nodes = num_of_nodes;
    vector<string> a_node(num_nodes,"a");
    vector<vector<int>> a_adj;
    for (int i=0; i<num_nodes; i++){
        vector<int> tmp;
        if (i == 0){
            for (int j=1;j<num_nodes;j+=4){
                tmp.push_back(j);
                if (j+1<num_nodes){
                    tmp.push_back(j+1);
                }
            }
            a_adj.push_back(tmp);
            tmp.clear();
        } else if (i % 4 == 2){
            if (i+1<num_nodes){
                tmp.push_back(i+1);
            }
            if (i+2<num_nodes){
                tmp.push_back(i+2);
            }
            a_adj.push_back(tmp);
            tmp.clear();
        }else{
            a_adj.push_back(tmp);
            tmp.clear();
        }
    }


    vector<string> b_node(num_nodes,"b");
    vector<vector<int>> b_adj;
    for (int i=0;i<num_nodes; i++){
        vector<int> tmp;
        if (i==0){
            for (int j=1;j<num_nodes; j+=4){
                tmp.push_back(j);
                if (j+3<num_nodes){
                    tmp.push_back(j+3);
                }
            }
            b_adj.push_back(tmp);
            tmp.clear();
        } else if (i%4 == 1){
            if (i+1<num_nodes){
                tmp.push_back(i+1);
            }
            if (i+2<num_nodes){
                tmp.push_back(i+2);
            }
            b_adj.push_back(tmp);
            tmp.clear();
        } else{
            b_adj.push_back(tmp);
            tmp.clear();
        }
    }




    int f = standard_ted(a_node, a_adj, b_node, b_adj,num_threads,parallel_version);
    cout << endl;
    printf("The final distance is %d\n",f);

}

