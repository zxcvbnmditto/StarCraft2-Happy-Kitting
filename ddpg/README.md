## Overview
### Actor
1. Input layer: size = 11
```
all player unit's: x coordinate, y coordinates, hp, distance to the closest enemy, and hp, selected

all opponent unit's: x coordinate, y coordinates, hp
```
2. Output layer: size = 5
```
attact enemy
move up
move down
move left
move right
```
3. Hidden layer:
```
1 hidden layer
8 nodes
Activation function = sigmoid + softmax
Batch Normalization = False
```

### Critic
1. Input layer: size = 16 * batch size
```
all player unit's: x coordinate, y coordinates, hp, distance to the closest enemy, and hp, selected

all opponent unit's: x coordinate, y coordinates, hp

actions
```

2. Output layer: size = 5
```
attact enemy
move up
move down
move left
move right
```

3. Hidden layer: None

### Memory:
```
max: 5000 transitions
batch size = 32
```

### Scripted Contraints
1. The agent will move toward to enemy's once a epsidoes begins until any of it's unit is within the distance of 20 to an enemy
2. The agent will always select the unit that is closest to the enemy

### Important global variables
1. Change the following 3 flags to run the program successfully (main.py)
```
# specifies the agent file that you are running
flags.DEFINE_string("agent", "ddpg_agent.SmartAgent",
                    "Which agent to run")

# specifies the map that you want to test on 
flags.DEFINE_string("map", 'HK2V1', "Name of a map to use.")

# specifies the maximum step count for this run
flags.DEFINE_integer("max_agent_steps", 50000, "Total agent steps.")
```
2.  Change the option of save models, load models and save_pic to save and load the program properly (main.py)
```
LOAD_MODEL = True # true => loads the pretrained model from models/
SAVE_MODEL = True # true => save the model at models/ before the run terminates
SAVE_PIC = True # save the generated graphs ar graphs/
```

3. Change the important map info (in ddpg_agent.py)
 ```
DEFAULT_ENEMY_COUNT # set to 2 as default
DEFAULT_PLAYER_COUNT # set to 1 as default
ENEMY_MAX_HP # set to 150 as default
PLAYER_MAX_HP # set to 60 as default
```

### Show the network paramaters 
Run this after you have already run the agent at least once
```
tensorboard --logdir logs
```

## Result
In 500000 steps

Winning rate: 0.27%

Average leftover sum of enemys' hp  = 49.275

<p align="center">
  Leftover enemy hp in each episodes <br>
  <img src="graphs/eval.png">
</p>


<p align="center">
  Overview of change of enemy hp throughout 500000 steps <br>
  <img src="graphs/enemy_hp.png">
</p>
