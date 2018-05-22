#### About
This is a project of creating a Starcraft 2 bot applying reinforcement learning <br>

dqn_old is the older version of the work <br>
dqn_new is the latest version of the work

All the mini maps used in this project can be found in the maps folder <br>

#### Environment:
>Python 3 <br>
>Tensorflow 1.7.0

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
#### Modified the code
1. Change the flags in main.py if needed
2. Change the option of restore model, load model, and plot in main.py if needed
3. Change the important map info in the agent file dqn_new.py
```
DEFAULT_ENEMY_COUNT
DEFAULT_PLAYER_COUNT
ENEMY_MAX_HP 
PLAYER_MAX_HP
```

#### Current Progress:
1. Using the updated pysc2 API
2. Extract units hp and location as features
3. Caculate the minimum distance between each of the player's unit to the opponent's units

#### Coming up:
1. Test the performance of the agent and update the action, reward, and extracted features if needed
2. Training smart_agent3
3. Apply Batch Normalization & modified the DQN network structure


