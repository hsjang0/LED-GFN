'''
  Run experiment with wandb logging.

  Usage:
  python runexpwb.py --setting bag

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''
import argparse

import torch
import wandb
import options
from attrdict import AttrDict

from exps.bag import bag
from exps.tfbind8 import tfbind8_oracle
from exps.tfbind10 import tfbind10
from exps.qm9str import qm9str
from exps.sehstr import sehstr
# from exps.gfp import gfp
# from exps.utr import utr
from exps.rna import rna

setting_calls = {
  'bag': lambda args: bag.eval(args),
  'tfbind8': lambda args: tfbind8_oracle.eval(args),
  'tfbind10': lambda args: tfbind10.eval(args),
  'qm9str': lambda args: qm9str.eval(args),
  'sehstr': lambda args: sehstr.eval(args),
  # 'gfp': lambda args: gfp.eval(args),
  # 'utr': lambda args: utr.eval(args),
  'rna': lambda args: rna.eval(args),
}

def eval(args):
  # print(f'Using {args.setting=} ...')
  print(f'Using args.setting={args.setting} ...')
  exp_f = setting_calls[args.setting]
  exp_f(args)
  return

    
if __name__ == "__main__":
    args = options.parse_args()

     # RNA Binding - 4 different tasks
    if args.setting == "rna":
      args.saved_models_dir = f"{args.saved_models_dir}/L{args.rna_length}_RNA{args.rna_task}/" 
      wandb.init(project=f"{args.wandb_project}-L{args.rna_length}-{args.rna_task}",
                entity=args.wandb_entity,
                config=args,
                mode=args.wandb_mode)
    else:
      wandb.init(project=args.wandb_project,
                entity=args.wandb_entity,
                config=args, 
                mode=args.wandb_mode)
    args = AttrDict(wandb.config)
    # args.run_name = wandb.run.name if wandb.run.name else 'None'
    run_name = args.model
    if args.offline_select == "prt":
      run_name += "_" + args.offline_select
    
    if args.sa_or_ssr == "ssr":
      run_name += "_" + args.sa_or_ssr

    if args.mcmc == True:
      run_name += "_" + "mcmc"
      if args.mh == True:
        run_name += "_" + "mh"
    
    args.run_name = run_name.upper()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    args.device = device
    
    eval(args)
    
    
    
    
