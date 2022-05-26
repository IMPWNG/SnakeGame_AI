import torch
import random
import numpy as np
from collections import deque
from snakegame import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # control the randomness of the agent
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # If we exed the memory, it will automatically train remove element from the left -> popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, snakegame):
        head = snakegame.snake[0]
        
        #Create 4 point arround the head - 20 is for the bloc size on snakegame.py
        point_l = Point(head.x -20, head.y)
        point_r = Point(head.x +20, head.y)
        point_u = Point(head.x, head.y -20)
        point_d = Point(head.x, head.y +20)
        
        # Boolean to know if the point is in the snake -> only one set to TRUE=1 other to FALSE=0
        dir_l = snakegame.direction == Direction.LEFT
        dir_r = snakegame.direction == Direction.RIGHT
        dir_u = snakegame.direction == Direction.UP
        dir_d = snakegame.direction == Direction.DOWN
        
        state = [
            #Dange straigh ahead
            (dir_r and snakegame.is_collision(point_r)) or 
            (dir_l and snakegame.is_collision(point_l)) or 
            (dir_u and snakegame.is_collision(point_u)) or
            (dir_d and snakegame.is_collision(point_d)),
            
            #Danger on the right
            (dir_d and snakegame.is_collision(point_r)) or
            (dir_u and snakegame.is_collision(point_l)) or
            (dir_r and snakegame.is_collision(point_u)) or
            (dir_l and snakegame.is_collision(point_d)),
            
            #Danger on the left
            (dir_d and snakegame.is_collision(point_r)) or
            (dir_u and snakegame.is_collision(point_l)) or
            (dir_r and snakegame.is_collision(point_u)) or
            (dir_l and snakegame.is_collision(point_d)),
            
            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Food location
            snakegame.food.x < snakegame.head.x, #food on the left
            snakegame.food.x > snakegame.head.x, #food on the right
            snakegame.food.y < snakegame.head.y, #food on the top
            snakegame.food.y > snakegame.head.y #food on the bottom
        ]
        
        return np.array(state, dtype=int)
        
        
    def remember(self, state, action, reward, next_state, done):
        # If it exced the maxumum memory, it will remove the oldest element -> popleft()
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # Return a list of tuples (a finite ordered list (sequence) of elements.)
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # Random action : tradeoff between randomness and exploitation
        self.epsilon = 80 - self.n_games # More game played = smaller epsilon (epsilon = how many games played)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snakegame = SnakeGameAI()
    while True:
        # get the old state of the current state
        state_old = agent.get_state(snakegame)
        
        #Get moove of the final state
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = snakegame.play_step(final_move)
        state_new = agent.get_state(snakegame)
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember the old state, action, reward, new state, done
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory + plot the result
            snakegame.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('game', agent.n_games, 'score', score, 'record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
                
if __name__ == '__main__':
    train()