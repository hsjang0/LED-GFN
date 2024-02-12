# Synthesis of MCMC and GFlowNet

Please first install your conda with yaml file and install the pyg (of pytorch 1.13). 
(https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)


### Setting

We highly recommend reading the paper first: "Towards Understanding and Improving GFlowNet Training".

There are multiple MDP implementations, but we follow prepend-append MDP (can generate a sequence with prepend or append). 

We mainly use SSR (S x S -> R) parameterization rather than SA (S-> A) parameterization for policy modeling. 

If you want to use SA "--sa_or_ssr sa," then you must modify some codes (TB is okay, but DB and SubTB are not designed for SA).

### Code references
Our implementation is based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet). 

### Our contribution (in terms of codes)

We updated RNA-binding tasks, detailed balance (DB), sub-trajectory balance (Sub-TB), and our method of MCMC-GFN. 

### Large files
Large files `sehstr_gbtr_allpreds.pkl.gz` and `block_18_stop6.pkl.gz` are available for download at https://figshare.com/articles/dataset/sEH_dataset_for_GFlowNet_/22806671
DOI: 10.6084/m9.figshare.22806671
These files should be placed in `datasets/sehstr/`.


### How to run?


You can fix the setting at 'exp/[task]/setting.yaml'

You can run your experiment by 


#### Run detailed balance
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model db --offline_select random
```


#### Run sub-trajectory balance
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model subtb --lamda 0.9 --offline_select random
```


#### Run trajectory balance
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model tb --offline_select random
```


#### Run trajectory balance with prioritized replay training
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model tb --offline_select prt
```

#### Run Sub-GFN (PRT+SSR+SUB)

```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model sub --offline_select prt
```

#### Run trajectory balance with filtering-based MCMC (ours)
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model tb --offline_select prt --mcmc True
```

#### Run trajectory balance with metropolis-hasting (MH) based MCMC (ours)
```
python runexpwb.py --setting tfbind8 --sa_or_ssr ssr --model tb --offline_select prt --mcmc True --mh True
```


