#### About
This is a project of creating a Starcraft 2 bot applying reinforcement learning

You can find three agents applied with three different algorithm: DQN, DDPG, and NEAT.

I set the inputs to be a list of features of length 11, which contains the information of player units' hp, coordinates, closest distance to the opponent's units and opponents hp and coordinates. (I know its a bit confusing) 

The outputs are 6 actions: attack, move up, move down, move right, move left, and select unit. 

The dqn and ddpg models are pretrained with 25000 steps, and the neat model is pretrained with 10 generations. 

All the mini maps used in this project can be found in the maps folder <br>
You can also find some pretrained models in the models folder


#### Environment:
>Python 3 

>Tensorflow 1.8.0

>Neat-python 0.92

#### Installation
1. Clone this repository
```
git clone https://github.com/zxcvbnmditto/StarCraft2-Happy-Kitting.git
```
2. Install pysc2 manually (pysc2 just got updated so pip3 doesn't give you the latest version yet) 
```
Find the folder where you python3 package is and download/replace the pysc2 folder from 

https://github.com/deepmind/pysc2
```
3. Install the required packages if needed using pip
```
pip3 install <package-name>
```
4. Move the HappyKiting3V2.SC2Map to the Starcraft 2 minimap folder.
```
Application/StarCraft II/Maps/mini_games (MacOS)
~/StarCraft II/Maps/mini_games (Linux)
```
#### Running
1. Lastly, go to the directory where u clone this repository and run
```
 python3 -m main
```
#### Modified the code for ddpg or dqn
1. Change the flags in main.py if needed
2. Change the option of save model, load model, and save_pic in main.py if needed
3. Change the important map info in the agent file
```
DEFAULT_ENEMY_COUNT
DEFAULT_PLAYER_COUNT
ENEMY_MAX_HP 
PLAYER_MAX_HP
```

#### Use Tensorboard to visualize the data for ddpg or dqn
```
Run this after you have already run the agent at least once
tensorboard --logdir logs/
```
#### Modified the code for neat
1. Change the important flags in main.py if needed
2. Change these global variables variables to control the running time in main.py
```
Here are some most important variables that you should keep an eye on

GENERATION = 11 # num of generation
pop # num of genomes for each generation(this is in config)
GENERATION_EP = 5 # evaluate each genome by average 10-episode rewards
EP_STEP = 500  # maximum episode steps
```
3. Save, load, evaluation
```
Evaluation generates the network graph

TRAINING # training or evaluating
CONTINUE_TRAINING # Train from scratch or from previous checkpoints
CHECKPOINT # this number is used to specified where the saved neat model should be restored for evaluation or continue training
```
4. You may also change the parameters in config file (If u know the effect)
5. Currently, the checkpoint file is set to be saved automatically after each generation. You can modify it by changing the number of this line
```
pop.add_reporter(neat.Checkpointer(<num-can-be-changed>))
```
6. Fitness function in main.py
7. Reward in agent.py

#### Current Progress:
1. Using the updated pysc2 API
2. Extract units hp and location as features
3. Caculate the minimum distance between each of the player's unit to the opponent's units
4. Use disabled actions to filter the unnecessary action based on current state
5. Save, restore model, and plotting data enabled
6. Prescipted a few initial actions of the agent to increase the performance by not wasting steps searching the enemy cluelessly
7. Temporarily fix and generalize the select_point action
8. Generalize the code so it is easy to be modified and tested
9. Applied Tensorboard to dqn and modified the DQN network structure


#### Coming up:
1. Test the performance of the agent and update the action, reward, and extracted features if needed
2. Applied tensorboard to the ddpg network and potentially modified the ddpg network
3. Tweaking :)

#### Reminder
1. Neat is still under the process of developing. It may have some bugs not being fixed yet
2. I will have to double check the ddpg algorithm implemenation, especially the a_loss and gradients

#### Results
We will evaluate our result using the enemy hp