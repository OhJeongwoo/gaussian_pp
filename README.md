experiment schedule

-- summary deadline         : 2021.06.11
-- presentation deadline    : 2021.06.16
-- final report deadline    : 2021.06.18


1. eps run-up validation test

1) in metaworld push-v2 environment
2021.06.08 16:00 finished

2) gaussian random path visualization
2021.06.07 23:00 finished

2. GRP algorithm hyperparameter tuning (in push-v1 environment)
- use entropy loss : True
- use eps runup : True

- N (# of anchors) : 5, 10, 20
- T (# of predictions) : 3, 5, 7

 T   3   5   7
N

5   AA  AB  AC

10  BA  BB  BC

20  CA  CB  CC


       COM           START                END

AA   local02    2021.06.08 16:30    2021.06.09 09:00 (expected)   
AB   local02    2021.06.08 16:30    2021.06.09 03:00 (expected)   
AC   cpslab4    2021.06.08 17:00    2021.06.09 03:00 (expected)    
BA   cpslab4    2021.06.08 17:00    2021.06.09 15:30 (expected)    
BB    navi2     2021.06.08 19:00    2021.06.09 06:00 (expected)   
BC    navi2     2021.06.08 19:00    2021.06.09 03:30 (expected)   
CA   local02    2021.06.09 07:00    2021.06.09 22:00 (expected)   
CB   local02    2021.06.09 07:00    2021.06.09 22:00 (expected)   
CC   cpslab4    


3. GRP-imitation
1) expert data collection (local01) (2021.06.08)
with ppo algorithm in 3 environments (push-v1, soccer-v1, faucet-v1)

data structure
- state seq
- action seq
- reward seq

2) pre-training with expert data (local01)
-> 06.09


4. baseline learning
- NDP
- PPO-multi
- GRP
- GRP-imitation

3 environments - push, soccer, faucet

   algo      NDP       PPO-multi    GRP     GRP-imitation
env

push         AA         AB          AC          AD

soccer       BA         BB          BC          BD

faucet       CA         CB          CC          CD


       
       COM           START                END

AA    raichu    2021.06.08 15:00    
AB    raichu    2021.06.08 15:00    2021.06.09 00:30
AC    
AD
BA    raichu     
BB    raichu    2021.06.09 00:30    2021.06.09 05:00
BC
BD
CA    raichu    
CB    raichu    
CC
CD


