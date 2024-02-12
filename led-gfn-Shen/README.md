# Note

The code is based on [https://github.com/maxwshen/gflownet](https://github.com/maxwshen/gflownet)

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


