import random, time
import pickle
import numpy as np
import torch
import wandb
from tqdm import tqdm
# import ray
import gc
import random
from . import guide
from .data import Experience


class Trainer:
  def __init__(self, args, model, mdp, actor, monitor):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.actor = actor
    self.monitor = monitor

  def learn(self, *args, **kwargs):
    print(f'Learning without guide workers ...')
    self.learn_default(*args, **kwargs)

  def handle_init_dataset(self, initial_XtoR):
    if initial_XtoR:
      print(f'Using initial dataset of size {len(initial_XtoR)}. \
              Skipping first online round ...')
      if self.args.init_logz:
        self.model.init_logz(np.log(sum(initial_XtoR.values())))
    else:
      print(f'No initial dataset used')
    return

  """
    Training
  """
  def learn_default(self, initial_XtoR=None, ground_truth=None):
    allXtoR = initial_XtoR if initial_XtoR else dict()
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    print(f'Starting active learning. \
            Each round: num_online={num_online}, num_offline={num_offline}')
    
    total_samples = []
    limited_buffer = []
    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      # Online training - skip first if initial dataset was provided
      if not initial_XtoR or round_num > 0:
        for _ in range(num_online):
          
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                  epsilon=self.args.explore_epsilon)
            
            # Log samples
            self.monitor.log_samples(round_num, explore_data)
            
            # Save to buffer for LED
            limited_buffer.append(explore_data)
            limited_buffer = limited_buffer[-100:]
              
          # Save to full dataset
          for exp in explore_data:
            if exp.x not in allXtoR:
              allXtoR[exp.x] = exp.r              
 

          for step_num in range(self.args.num_steps_per_batch):
            # Learning energy decomposition
            if (self.args.model in ['subtb_rd','db_rd']):
              for _ in range(self.args.led_step):
                self.model.train_proxy(limited_buffer[random.randint(0,len(limited_buffer)-1)])
            self.model.train(explore_data)          
          
          if self.args.model in ['ppo']:
            for step_num in range(self.args.num_steps_per_batch):
              self.model.train(explore_data)
      
      
      if self.args.model not in ['ppo']:
        for _ in range(num_offline):
          with torch.no_grad():
            offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
            offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)

            # Save to buffer for LED
            limited_buffer.append(offline_dataset)
            limited_buffer = limited_buffer[-100:]
          
          for step_num in range(self.args.num_steps_per_batch):
            # Learning energy decomposition
            if (self.args.model in ['subtb_rd','db_rd']):
              for _ in range(self.args.led_step):
                self.model.train_proxy(limited_buffer[random.randint(0,len(limited_buffer)-1)])
            self.model.train(offline_dataset)
       
       
      if round_num % monitor_fast_every == 0 and round_num > 0:
        self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
      
      """
      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + "/" + f'{wandb.run.id}_round_{round_num}.pth')
        with open(self.args.saved_models_dir + \
                  self.args.run_name + "/" + f"{wandb.run.id}_round_{round_num}_sample.pkl", "wb") as f:
          pickle.dump(total_samples, f)
      """
      
    print('Finished training.')
    return


  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    select = self.args.get('offline_select', 'prt')
    if select == 'prt':
      return self.__biased_sample_xs(allXtoR, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)


  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []

    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def offline_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """

    offline_rs = [allXtoR[x] for x in offline_xs]

    # Not subgfn: sample trajectories from backward policy
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    offline_trajs = [
      Experience(traj=traj, x=x, r=r,
                logr=torch.log(torch.tensor(r, dtype=torch.float32,device=self.args.device))
                )
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    return offline_trajs

 
