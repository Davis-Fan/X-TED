#include "TED_C++.h"

int ToInt(const string &str){
    return stoi(str);
}

vector<int> To_Vector(string str) {
    vector<int> vec_int;
    string word = "";
    str.erase(0, 1);
    str.pop_back();
    if (str.size() == 0){
        return vec_int;

    }else{

        for (int i = 0; i < str.size(); i++) {
            if ((str[i] != ' ') && (str[i] != ',')) {
                word += str[i];

                if(i == str.size()-1){
                    int k = ToInt(word);
                    vec_int.push_back(k);
                    word = "";
                }

            }
            else if (str[i] == ','){
                int k = ToInt(word);
                vec_int.push_back(k);
                word = "";
            }
        }
        return vec_int;
    }
}

vector<string> splitting(string str){
    vector<string> split;
    string part = "";
    str.erase(0, 1);
    str.pop_back();
    for (int i = 0; i < str.size(); i++){
        if(str[i] == '['){
            int k = i;
            while(str[k] != ']'){
                part = part + str[k];
                k++;
            }
            part = part + ']';
            split.push_back(part);
            part = "";
        }
    }
    return split;
}

// This is just preprocessing for dataset formatting not the preprocessing algorithm we proposed
vector<vector<int>> pre_process(string str){
    vector<vector<int>> two_D_vector;
    vector<string> partition = splitting(str);
    for (int i = 0; i < partition.size(); i++){
        vector<int> t = To_Vector(partition[i]);
        two_D_vector.push_back(t);
    }
    return two_D_vector;
}

vector<string> node_process(string str){
    vector<string> node;
    string part = "";
    for (int i = 0; i < str.size(); i++){
        if (str[i] != ' '){
            part = part + str[i];
        }else{
            node.push_back(part);
            part = "";
        }
    }

    return node;
}

