# PPO-Inverted-Double-Pendulum
Simple implementation of proximal policy optimization on Gymnasium's Inverted Double Pendulum environement. Through training, the agent learns to balance the inverted double pendulum with a moving cart, while PPO objective clipping and KL early stopping insure stability.

## Dependencies
PyTorch, Matplotlib, Gymnasium

## Usage
Run training.py to train the model on the inverted double pendulum environment. This will create a new folder called "data" containing saved models and performance graphs. Training length can be adjusted in training.py, and hyperparameters can be adjusted in agent.py. When you are satisfied with the model's performance, run inference.py to showcase it.