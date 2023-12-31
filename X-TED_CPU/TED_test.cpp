#include "TED_C++.h"

void test(int num_threads, int parallel_version,  char* input_1, char* input_2){

    ifstream in(input_1);
    string line = "";
    vector<string> nodes;
    if(in){
        while (getline (in, line)){
            nodes.push_back(line);
            line = "";
        }
    }

    ifstream in_adj(input_2);
    string line_2 = "";
    vector<string> nodes_adj;
    if(in_adj){
        while (getline (in_adj, line_2)){
            nodes_adj.push_back(line_2);
            line_2 = "";
        }
    }

    int dimension = nodes.size();
    vector<vector<int>> Distance(dimension, vector<int>(dimension));

    for (int i = 0; i<dimension; i++){
        for (int j = 0; j<dimension; j++){

            if (i==0 && j== 1) {

                string node_1 = nodes[i];
                string node_1_adj = nodes_adj[i];
                string node_2 = nodes[j];
                string node_2_adj = nodes_adj[j];

                vector<string> node1 = node_process(node_1);
                vector<vector<int>> node1_adj = pre_process(node_1_adj);
                vector<string> node2 = node_process(node_2);
                vector<vector<int>> node2_adj = pre_process(node_2_adj);

                int d = standard_ted(node1, node1_adj, node2, node2_adj, num_threads, parallel_version);
                Distance[i][j] = d;
                cout << "The distance = " << d << endl;
            }
        }
    }

}