#### About
This is a project of creating a Starcraft 2 bot applying reinforcement learning <br>

smart_agent.py is the base structure of all agents <br>
smart_agent2.py is designed to run the 3V2 map <br>
smart_agent3.py is designed to run the 2V1 map <br>

All the mini maps used in this project can be found in the maps folder <br>

#### Environment:
>Python 3 <br>
>Tensorflow 1.7.0

#### Installation
1. Clone this repository
```
git clone https://github.com/zxcvbnmditto/StarCraft2-Happy-Kitting.git
```
2. Install the required packages if needed using pip
```
pip3 install <package-name>
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
3. Apply Batch Normalization & modified the DQN network structure
