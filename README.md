# StarCraft2-Happy-Kitting

#### About the project
StarCraft2 A.I. implemented with Deep Deterministic Policy Gradient Algorithm (DDPG)

smart_agent.py is implemented with q-learning algorithm <br>
smart_agent2.py is implemented with DQN algorithm
smart_agent3.py is implemented with DQN using the updated pysc2 API

#### Environment:
>Python 3 <br>
>Tensorflow 1.7.0

#### Installation
1. Clone this repository
>git clone https://github.com/zxcvbnmditto/StarCraft2-Happy-Kitting.git
2. Install the required packages if needed
> pip3 install tensorflow <br>
> pip3 install absl-py <br>
> pip3 install numpy <br>
> pip3 install pandas <br>
> pip3 install pysc2 <br>
3. Move the HappyKiting3V2.SC2Map to the Starcraft 2 minimap folder. If you are a Mac user, the directory should look similar to something like this
> /Application/StarCraft II/Maps/mini_games
4. Lastly, go to the directory where u clone this repository and run 
'''
 python3 -m main
'''
#### Current Progress:
1. Using the updated pysc2 API
2. Extract units hp and location as features
3. Caculate the minimum distance between each of the player's unit to the opponent's units


#### Coming up:
1. Test the performance of the agent and update the action, reward, and extracted features if needed
2. Training smart_agent3

#### Environment:
>Python 3 <br>
>Tensorflow 1.7.0

#### Installation
1. Clone this repository
>git clone https://github.com/zxcvbnmditto/StarCraft2-Happy-Kitting.git
2. Install the required packages if needed using pip
```
pip install <package-name>
```
3. Move the HappyKiting3V2.SC2Map to the Starcraft 2 minimap folder. 
```
Application/StarCraft II/Maps/mini_games (MacOS)
~/StarCraft II/Maps/mini_games (Linux)
```
4. Lastly, go to the directory where u clone this repository and run
```
 python3 -m main
```
#### Current Progress:
1. Using the updated pysc2 API
2. Extract units hp and location as features
3. Caculate the minimum distance between each of the player's unit to the opponent's units

#### Coming up:
1. Test the performance of the agent and update the action, reward, and extracted features if needed
2. Training smart_agent3

