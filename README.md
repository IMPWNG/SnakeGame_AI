# Snake Game with AI - Reinforcement Learning

- Step 1:
    - Setup the environment + import the snake game
- Step 2:
    - Remove the Q-table and lets play by itself
    - Create the Q-learning algorithm -> Agent (Q-learning)

## Focus on this one for the randomes of moves and when the randomnes transform exploitation

````
    def get_action(self, state):
        # Random action : tradeoff between randomness and exploitation
        self.epsilon = 80 - self.n_games # More game played = smaller epsilon (epsilon = how many games played)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #
            move = random.randint(0,3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move - torch.argmax(prediction).item()
            final_move[move] = 1
````

## DONE ✅

### Futur improvements

- Host on a flask App
- Add a GUI
