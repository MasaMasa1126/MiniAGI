#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiniAGI - Further Refined with GAE PPO
---------------------------------------------------

特徴:
1. PPO ActorCriticに GAE(Generalized Advantage Estimation) を導入し、学習を安定化。
2. EpisodicMemory を用いて複数ステップの経験を蓄え、一括でPPO更新(clip + GAE)。
3. マルチモーダル(視覚25 + テキスト1) MultiheadAttentionはembedding次元を大きめ(64)に。
4. シンボリック推論 (BFS?3 => approach推奨 => 行動が距離短縮したら+0.05報酬) を継続し、Neuro-Symbolicシナジーを示す。
5. Curiosity, Meta, GlobalWorkspace, RealWorldIF など継続。Ablationで比較。

実行:
    pip install torch matplotlib
    python miniagi_final_ver6.py
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from queue import Queue

###############################################################################
# 1. DynamicGlobalWorkspace
###############################################################################
class DynamicGlobalWorkspace:
    def __init__(self):
        self.active_content = None
        self.modules = []
        self.base_threshold = 0.7
        self.dynamic_offset = 0.0
        self.history = deque(maxlen=10)

    def register_module(self, module):
        self.modules.append(module)

    def broadcast(self, content, source_module, priority):
        thr = self.base_threshold + self.dynamic_offset
        if priority>thr:
            if self.active_content is not None:
                self.history.append(self.active_content)
            self.active_content = content

    def adjust_threshold(self, delta):
        self.dynamic_offset += delta
        self.dynamic_offset = max(min(self.dynamic_offset,0.3),-0.3)

    def get_active_content(self):
        return self.active_content
    def get_history(self):
        return list(self.history)

###############################################################################
# 2. SymbolicReasoningModule
###############################################################################
class SymbolicReasoningModule:
    def __init__(self):
        self.knowledge_graph = {}
        self.weighted_rules  = []
        self.accept_threshold= 0.5

    def add_weighted_rule(self, confidence, condition, consequence):
        self.weighted_rules.append((confidence, condition, consequence))

    def infer(self, facts):
        new_facts=[]
        max_iter=5
        for _ in range(max_iter):
            found_new=False
            for (conf,cond,cons) in self.weighted_rules:
                if all(c in facts for c in cond):
                    if conf> self.accept_threshold:
                        for c2 in cons:
                            if (c2 not in facts) and (c2 not in new_facts):
                                new_facts.append(c2)
                                found_new=True
            if not found_new:
                break
            facts= facts+ new_facts
        return new_facts

###############################################################################
# 3. MetaReasoningModule
###############################################################################
class MetaReasoningModule:
    def __init__(self):
        pass

    def process(self, symbolic_module, global_workspace):
        nr= len(symbolic_module.weighted_rules)
        if nr>8:
            global_workspace.adjust_threshold(+0.01)
        elif nr<5:
            global_workspace.adjust_threshold(-0.01)

###############################################################################
# 4. CuriosityModule
###############################################################################
class CuriosityModule:
    def __init__(self, scale=0.05):
        self.scale= scale
        self.prev_state=None

    def compute_reward(self, cur_vec):
        if self.prev_state is None:
            self.prev_state= cur_vec.copy()
            return 0.0
        diff= np.abs(cur_vec- self.prev_state).mean()
        self.prev_state= cur_vec.copy()
        return diff * self.scale

###############################################################################
# 5. RealWorldInterface
###############################################################################
class RealWorldInterface:
    def __init__(self):
        pass
    def process_sensors(self): pass
    def send_commands(self,cmds): pass

###############################################################################
# 6. LargeGridPOEnvV2 (5x5 partial obs) 
###############################################################################
class LargeGridPOEnvV2:
    """
    10x10 ランダム + 部分可視(5x5) + BFS距離
    synergy: BFSdist <=3 => Symbolic => approach => shaped reward
    """
    def __init__(self, size=10, wall_prob=0.15):
        self.size=size
        self.wall_prob= wall_prob
        self.max_steps=100
        self.reset()

    def _random_map(self):
        g=[]
        for r in range(self.size):
            row=[]
            for c in range(self.size):
                if random.random()<self.wall_prob: row.append("#")
                else: row.append(".")
            g.append(row)
        return g

    def _in_bounds(self,r,c):
        return (0<=r<self.size and 0<=c<self.size)

    def _bfs_distance(self,gr,gc,grid):
        dist= [[float('inf')]*self.size for _ in range(self.size)]
        Q=Queue()
        Q.put((gr,gc))
        dist[gr][gc]=0
        while not Q.empty():
            rr,cc= Q.get()
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc= rr+dr, cc+dc
                if self._in_bounds(nr,nc) and grid[nr][nc]!="#":
                    if dist[nr][nc]> dist[rr][cc]+1:
                        dist[nr][nc]= dist[rr][cc]+1
                        Q.put((nr,nc))
        return dist

    def reset(self):
        while True:
            grid= self._random_map()
            sr,sc= random.randrange(self.size), random.randrange(self.size)
            gr,gc= random.randrange(self.size), random.randrange(self.size)
            if grid[sr][sc]=="#" or grid[gr][gc]=="#":
                continue
            grid[sr][sc]="S"
            grid[gr][gc]="G"
            dist_map= self._bfs_distance(gr,gc,grid)
            if dist_map[sr][sc]< float('inf'):
                self.grid= grid
                self.start= (sr,sc)
                self.goal= (gr,gc)
                self.dist_map= dist_map
                break
        self.agent_pos= self.start
        self.steps=0
        return self._get_obs()

    def step(self,action):
        r,c= self.agent_pos
        if action==0: nr,nc= r-1,c
        elif action==1: nr,nc= r+1,c
        elif action==2: nr,nc= r,c-1
        else: nr,nc= r,c+1
        if not self._in_bounds(nr,nc) or self.grid[nr][nc]=="#":
            nr,nc=r,c
        prev_dist= self.dist_map[r][c]
        self.agent_pos=(nr,nc)
        self.steps+=1
        reward=0.0
        done=False
        if self.agent_pos==self.goal:
            reward=1.0
            done=True
        if self.steps>= self.max_steps:
            done=True
        # synergy -> (別で shaped rewardを計算)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # 5x5 partial
        r,c= self.agent_pos
        half=2
        obs_vision=[]
        for dr in range(-half,half+1):
            for dc in range(-half,half+1):
                rr,cc= r+dr, c+dc
                if not self._in_bounds(rr,cc): ch="#"
                else: ch=self.grid[rr][cc]
                obs_vision.append(ch)
        # obs_vision => 25 tokens
        v_tokens=[]
        for ch in obs_vision:
            if ch=="#": v_tokens.append(0)
            elif ch=="." or ch=="S": v_tokens.append(1)
            elif ch=="G": v_tokens.append(2)
            else: v_tokens.append(1)
        # BFS => text
        d= self.dist_map[r][c]
        if d==float('inf'): d=self.size*self.size
        t_token= d%32
        return (v_tokens, t_token)

###############################################################################
# 7. PPO + GAE 
###############################################################################
class GAEBuffer:
    """
    trajectory for 1 epoch
    """
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma= gamma
        self.lam= lam
        self.states=[]
        self.actions=[]
        self.logprobs=[]
        self.rewards=[]
        self.values=[]
        self.dones=[]
    def store(self, state, action, logprob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    def get_size(self):
        return len(self.states)

class PPOActorCritic(nn.Module):
    def __init__(self, vocab_size=3, text_vocab=32, embed_dim=64, nhead=4, hidden_dim=128, gamma=0.99, lam=0.95, clip_eps=0.1, lr=1e-3):
        super().__init__()
        self.gamma= gamma
        self.lam= lam
        self.clip_eps= clip_eps

        self.vision_embed= nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.text_embed  = nn.Embedding(num_embeddings=text_vocab, embedding_dim=embed_dim)
        self.pos_embed   = nn.Embedding(num_embeddings=26, embedding_dim=embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True)

        self.policy_head= nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.value_head= nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, vision_tokens, text_token):
        # vision_tokens: list of 25 int, text_token: int
        seq_len=25+1
        vs = torch.tensor(vision_tokens, dtype=torch.long).unsqueeze(0)
        ts = torch.tensor([text_token], dtype=torch.long).unsqueeze(0)
        vs_embed= self.vision_embed(vs)
        ts_embed= self.text_embed(ts)
        cat_embed= torch.cat([vs_embed, ts_embed], dim=1)

        pos_idx= torch.arange(seq_len).unsqueeze(0)
        pos_e= self.pos_embed(pos_idx)
        cat_embed= cat_embed+ pos_e

        out,_= self.attn(cat_embed, cat_embed, cat_embed)
        pooled= out.mean(dim=1)
        policy= self.policy_head(pooled)
        value = self.value_head(pooled)
        return policy, value

    def act(self, vision_tokens, text_token):
        policy, value= self.forward(vision_tokens, text_token)
        probs= nn.Softmax(dim=1)(policy)
        dist= torch.distributions.Categorical(probs)
        action= dist.sample()
        logprob= dist.log_prob(action)
        return action.item(), logprob.detach(), value[0,0].detach()

    def evaluate_actions(self, states, actions):
        """
        states: list of (v_tokens, t_token)
        actions: tensor
        => return new_logprobs, new_values, dist_entropy
        """
        bat_size= len(states)
        v_all= []
        for (v_tokens,t_token) in states:
            v_all.append((v_tokens,t_token))

        # バッチ処理(=1つずつembedding?):
        pol_list=[]
        val_list=[]
        for i in range(bat_size):
            v_tokens, t_token= v_all[i]
            p,v= self.forward(v_tokens, t_token)
            pol_list.append(p)
            val_list.append(v)
        logits= torch.stack(pol_list)  # (batch,4)
        values= torch.stack(val_list)  # (batch,1)
        probs= nn.Softmax(dim=1)(logits)
        dist= torch.distributions.Categorical(probs)
        new_logprobs= dist.log_prob(actions)
        ent= dist.entropy().mean()
        return new_logprobs, values.squeeze(1), ent

    def ppo_update(self, buffer):
        states= buffer.states
        actions= torch.tensor(buffer.actions,dtype=torch.long)
        rewards= buffer.rewards
        values= buffer.values
        dones= buffer.dones
        logprobs_old= torch.stack(buffer.logprobs)

        # compute GAE advantage
        advantages=[]
        gae=0.0
        values_ = values + [0.0]  # last value=0
        for i in reversed(range(len(rewards))):
            delta= rewards[i] + self.gamma * (0 if dones[i] else values_[i+1]) - values_[i]
            gae= delta + self.gamma*self.lam*(0 if dones[i] else 1)*gae
            advantages.insert(0,gae)
        advantages= torch.tensor(advantages,dtype=torch.float32)

        returns= advantages + torch.tensor(values,dtype=torch.float32)

        # evaluate
        new_logprobs, new_values, dist_entropy= self.evaluate_actions(states, actions)
        ratio= torch.exp(new_logprobs - logprobs_old)
        surr1= ratio*advantages
        surr2= torch.clamp(ratio,1.0-self.clip_eps,1.0+self.clip_eps)*advantages
        policy_loss= -torch.min(surr1,surr2).mean()
        value_loss= nn.MSELoss()(new_values, returns)
        loss= policy_loss + 0.5*value_loss - 0.01*dist_entropy

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


###############################################################################
# 8. CompleteMiniAGI Ver6
###############################################################################
class CompleteMiniAGI:
    """
    - 10x10, partial obs(5x5=25 tokens) + BFS( text 1 token ), synergy
    - PPO(GAE) with MultiheadAttention
    - Symbolic synergy => BFS<=3 => approach => if approach => shaped +0.05
    - ablation( curiosity, meta, symbolic )
    """
    def __init__(self, use_curiosity=True, use_meta=True, use_symbolic=True):
        self.gw= DynamicGlobalWorkspace()
        self.sym= SymbolicReasoningModule()
        self.meta= MetaReasoningModule()
        self.curiosity= CuriosityModule(scale=0.05) if use_curiosity else None
        self.policy= PPOActorCritic()
        self.rw_if= RealWorldInterface()

        self.env= LargeGridPOEnvV2(size=10, wall_prob=0.15)
        self.use_meta= use_meta
        self.use_symbolic= use_symbolic

        self.gw.register_module(self.sym)
        self.gw.register_module(self.meta)
        if self.use_symbolic:
            # BFS<=3 => approach => shaped
            self.sym.add_weighted_rule(
                0.8,
                [("dist_le3",)],
                [("action","recommend","approach")]
            )

        self.gamma= 0.99
        self.lam= 0.95
        self.buffer= GAEBuffer(gamma=self.gamma, lam=self.lam)

    def run_epoch(self, episodes=10):
        """
        1 epoch = multiple episodes => gather transitions => ppo update
        """
        total_reward= 0.0
        for ep in range(episodes):
            obs= self.env.reset()
            if self.curiosity:
                self.curiosity.prev_state= None

            done=False
            ep_reward=0.0
            while not done:
                if self.use_meta:
                    self.meta.process(self.sym, self.gw)

                (v_tokens, t_token)= obs
                distv= self.env.dist_map[self.env.agent_pos[0]][self.env.agent_pos[1]]
                # symbolic synergy
                facts=[]
                if distv<=3 and self.use_symbolic:
                    facts.append(("dist_le3",))
                new_facts= self.sym.infer(facts)

                # select action
                action, logprob_old, val_old= self.policy.act(v_tokens,t_token)
                before_d= distv
                next_obs, env_r, done, _= self.env.step(action)
                after_d= self.env.dist_map[self.env.agent_pos[0]][self.env.agent_pos[1]]

                shaped=0.0
                if ("dist_le3",) in facts and after_d< before_d and self.use_symbolic:
                    shaped=0.05

                c_r=0.0
                if self.curiosity:
                    # partial => 25 tokens => average?
                    merged_mean= np.mean(v_tokens)
                    c_r= self.curiosity.compute_reward(np.array([merged_mean, float(t_token)]))

                reward= env_r + shaped + c_r
                ep_reward+= reward

                self.buffer.store((v_tokens,t_token), action, logprob_old, val_old.item(), reward, done)
                obs= next_obs

            total_reward+= ep_reward

        # PPO update
        self.policy.ppo_update(self.buffer)
        self.buffer= GAEBuffer(gamma=self.gamma, lam=self.lam)

        return total_reward


###############################################################################
# 9. Main
###############################################################################
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    num_epochs= 50
    # ablation
    conds= [
        (False,False,False),
        (True,False,False),
        (False,True,True),
        (True,True,True)
    ]
    results={}
    labels=[]
    for (c_flag,m_flag,s_flag) in conds:
        label= f"C={c_flag},M={m_flag},S={s_flag}"
        agent= CompleteMiniAGI(use_curiosity=c_flag, use_meta=m_flag, use_symbolic=s_flag)
        ep_rews=[]
        for ep in range(num_epochs):
            r= agent.run_epoch(episodes=10)
            ep_rews.append(r)
        results[label]= ep_rews
        labels.append(label)

    plt.figure(figsize=(8,6))
    for lbl in labels:
        plt.plot(results[lbl], label=lbl)
    plt.title("MiniAGI Ver6: GAE PPO + MultiheadAttention + BFS synergy - Ablation")
    plt.xlabel("Epoch")
    plt.ylabel("SumReward(10ep)")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
