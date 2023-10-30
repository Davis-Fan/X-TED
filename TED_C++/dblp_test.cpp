#include "TED_C++.h"

void dblp_test(int num_threads, int parallel_version){
    ifstream in("/Users/davis/CLionProjects/TED_C++/pattern/dblp/different_nodes/1000_nodes/dblp_nodes_1000.txt");
//    ifstream in("/home/fan.1090/Davis/TED/TED_C++/bolzano_nodes.txt");
    string line = "";
    vector<string> nodes;
    if(in)
    {
        while (getline (in, line))
        {
            nodes.push_back(line);
            line = "";
        }
    }

    ifstream in_adj("/Users/davis/CLionProjects/TED_C++/pattern/dblp/different_nodes/1000_nodes/dblp_nodes_adj_1000.txt");
//    ifstream in_adj("/home/fan.1090/Davis/TED/TED_C++/bolzano_nodes_adj.txt");
    string line_2 = "";
    vector<string> nodes_adj;
    if(in_adj)
    {
        while (getline (in_adj, line_2))
        {
            nodes_adj.push_back(line_2);
            line_2 = "";
        }
    }


    int dimension = nodes.size();
    vector<vector<int>> Distance(dimension, vector<int>(dimension));
//    int Distance[9645][9645];
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
//            cout <<"i = "<< i << " j = " << j  << endl;
            }
        }
    }
    cout << "Total task finish, " << Total_time / 1000.0 << " ms consumed" << endl;


}

