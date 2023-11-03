# X-TED

This is the code repository for X-TED

---

## Supplementray of X-TED

The PDF file in this folder is the supplementray of X-TED, which contains the details of dynamic parallel strategy and its mathematics model.


---

## Download the code repository for X-TED

Our implementation of X-TED contains both CPU-version and GPU-version.

Firstly, use the command line below to download this repository.

```
$ git clone https://github.com/Davis-Fan/X-TED.git
$ cd X-TED
```

---

## Dataset

The complete version of the four dataset we used are uploaded in the link [X-TED Dataset](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/fan_1090_buckeyemail_osu_edu/EhXIR-JzOopIpw6KDA42Pn0BR5Z80VRh4Z9cGspeY7b8Cw?e=6zVslY). Due to the storage limit, we only show some randomly sampled cases in the folder **Sample_Dataset**, which contains trees from all datasets with nodes ranging from 100 nodes to 1000 nodes.

We use two files to store the labels of all nodes and the structure of trees respectively, based on the bracket format. For example, for a tree like below:

<pre>
Tree:     Preorder:
    a              0
   / \            / \
  b   c          1   2
     / \            / \
    d   e          3   4
</pre>

It can be represent by: {a{b}{c{d}{e}} in bracket format. It means that node *a* is the root and has two children *b* and *c*. And node *c* has two children *d* and *e*. Therefore, the labels of this tree is: [a, b, c, d, e]. And the tree structure can be store in a 2D list [ [1,2], [ ], [3,4], [ ], [ ] ].  The first index of this 2D list is the preorder of each node (in the preorder traversal); In this case, it's from 0 to 4. The element at index *i* is another list, which represents the preorder of all children nodes at this node *i*. For example, the first element, which is the root, it contains two nodes: 1 and 2, and therefore, the first element in the 2D list is [1,2]. For the second node, as it has no child node and hereby it is a empty list. 

In the test, we need to input two paths that contain the information of trees. For example, if we want to test a pair of 1000-nodes in the swissport dataset, the first parameter is the "Sampled_Dataset/1_Swissport/swissport_nodes_1000.txt " which includes the **labels** of all nodes for 1000-node trees (sampled) in the swissport. The second parameter is the "Sampled_Dataset/1_Swissport/swissport_nodes_adj_1000.txt" which includes the **structure** of all 1000-node trees (sampled) in the swissport.

---

## Run the X-TED

### Test GPU-verion

1) Enter the file named X-TED_GPU

```
$ cd X-TED_GPU
```

2) Create a new folder to contain the complied file

```
$ mkdir test
$ cd test
```

3) Complie code

```
$ cmake ..
$ make
```

4. Run the program. We need to input two paths that contain the information of trees. For example, if we want to test a pair of 1000-nodes in the swissport dataset, please use:

```
$ ./X-TED_GPU ../../Sampled_Dataset/1_Swissport/swissport_nodes_1000.txt ../../Sampled_Dataset/1_Swissport/swissport_nodes_adj_1000.txt
```

​		The program will execute the *same* test for 20 runs.

----------------------

### Test CPU-verion

Return to the file named X-TED_GPU

```
$ cd ..
$ cd ..
$ cd X-TED_CPU
```

Create a new folder to contain the complied file

```
$ mkdir test
$ cd test
```

Complie code

```
$ cmake ..
$ make
```

Run the program. Three input parameters are needed. The first parameter is the number of threads you want to use here and the other two are parameters. For example, if we want to test a pair of 1000-nodes in the swissport dataset by using 8-cores to compute, please use:

```
$ ./X-TED_CPU 8 ../../Sampled_Dataset/1_Swissport/swissport_nodes_1000.txt ../../Sampled_Dataset/1_Swissport/swissport_nodes_adj_1000.txt
```

Also, the program will execute the *same* test for 20 runs.
