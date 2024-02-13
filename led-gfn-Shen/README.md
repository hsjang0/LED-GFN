# Note

The code is based on [https://github.com/maxwshen/gflownet](https://github.com/maxwshen/gflownet)

### How to run?

You can fix the setting at 'exp/[task]/setting.yaml'

You can run your experiment by 

#### Run LED-GFN with subTB
```
python runexpwb.py --setting bag --sa_or_ssr ssr --offline_select random --model subtb_rd
```

#### Run LED-GFN with db
```
python runexpwb.py --setting bag --sa_or_ssr ssr --offline_select random --model db_rd
```
