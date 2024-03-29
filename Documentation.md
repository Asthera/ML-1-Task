# Documentation

Maked by Volodymyr Novokhatskyi

Task: [Mountain Car Continuous](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)

## Problem description

We have a car placed stochastically at the bottom of a sinusoidal valley, 
with the only possible actions being the accelerations that can be applied to the car in either direction.

The goal is to strategically accelerate the car to reach the goal state on top of the right hill.

### Action space

The action space is continuous and defined as a single float value in the range [-1, 1]. 
It represents the directional force applied on the car.

### Observation space

The observation space is continuous and defined by two float values.
The first value represents the position of the car, while the second value represents the velocity of the car. 
Position is in the range [-1.2, 0.6], while velocity is in the range [-0.07, 0.07].

### Library shapes

1. Observation space: (2,): 
    - `np.array([0.5, 0.05])`
2. Action space: (1,):
    - `np.array([0.5])`


### Environment dynamics


### Initial state

1. The position of the car is random value in the range [-0.6 , -0.4]. 
2. The velocity of the car is assigned to 0.

### Reward

A negative reward of -0.1 * (action_value) ^ 2 is received at each timestep to penalise for taking actions of large magnitude. 
If the mountain car reaches the goal then a positive reward of +100 is added to the negative reward for that timestep.

### Episode termination

The episode terminates if either of the following happens:

1. The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
2. The length of the episode is 999.


