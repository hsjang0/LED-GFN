# Note

The code is based on [https://github.com/maxwshen/gflownet](https://github.com/maxwshen/gflownet)

### Requirements

```bash
pip install -r pip_requirements.txt
```

### How to run?

You can fix the setting at 'exp/[task]/setting.yaml'

You can run your experiment by 

```
python runexpwb.py --setting bag --sa_or_ssr ssr --offline_select random --model subtb_rd or db_rd
```
