# Note

The code is based on [https://github.com/zdhNarsil/GFlowNet-CombOpt](https://github.com/zdhNarsil/GFlowNet-CombOpt)

### Requirements

```bash
pip install hydra-core==1.1.0 omegaconf submitit hydra-submitit-launcher
pip install dgl==0.6.1
```

### How to run?

You can fix the setting at 'configs/main.yaml'

You can run your experiment by 

```bash
python main.py --config-name main
```