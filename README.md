# StarCraft2-Happy-Kitting

#### About the project
StarCraft2 A.I. implemented with Deep Deterministic Policy Gradient Algorithm (DDPG)

smart_agent.py is implemented with q-learning algorithm <br>
smart_agent2.py is implemented with DQN algorithm

#### Environment:
>Python 3.6.5 <br>
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
> python3 -m main
#### Current Progress:
1. Manage to make the units to run in eight directions
2. Units can be selected individually after multi-select
3. Able to extract multiple data from the pysc2 API and used as input features

#### Coming up:
1. Modified the structure of the file system
2. Select the unit individually using select_point from the screen
3. Enhance the current reward assignment
4. Calculate the distance between the opponenent's units and ours


