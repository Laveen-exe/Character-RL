import os
import cv2
import torch
import random
import numpy as np
from collections import deque
from model_tr import Net, QTrainer
from neulab.Algorithms import ManhattanMetric


MAX_MEMORY = 100000000000
BATCH_SIZE = 10
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if memory fill call popleft()
        self.img = cv2.imread("blank.png",0)
        self.img = self.img.reshape(500,500,1)
        self.image = self.img.copy()
        self.target = cv2.imread("b.png",0)
        self.target = self.target.reshape(500, 500, 1)
        self.model = Net(input_size = (500,500,1), output_size = 5)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
        self.l = 0
        self.k = 0
        self.reward = 0
        self.delta = 0
        self.episode_delta = 0
        self.sum_mean = 0
        self.word = ''

    def get_state(self):
        state = cv2.imread("blank.png",0)
        state = state.reshape(500, 500, 1)
        return state


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            print("hello")
        else:
            print("bye")
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        for state, action, reward, next_state, done in mini_sample:
           self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves : tradeoff exploration / exploitation
        self.epsilon = 10 - self.n_games
        final_move = [0] * 5
        if random.randint(0, 20) < self.epsilon:
            move = random.randint(0,4)
            final_move[move] = 1
        else:
            state0 = state
            shape_img = state.shape
            state0 = state0.reshape(shape_img[2], shape_img[0], shape_img[1])
            state0 = torch.FloatTensor(state0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def play_step(self, action):
        ## Perform action

        index = action.index(1) + 65
        string = chr(int(index))
        self.word += string

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 5
        color = (0, 0, 0)
        thickness = 15
        self.k += 1

        print(self.word)
        print("L is :", self.l)
        if self.l == 2:
            done = True
            filename = 'helloworld/savedImage' + str(self.k) + '.png'
            cv2.imwrite(filename, self.image)
            self.image = self.img.copy()

            print("The sum of mean of Episode is : ", self.sum_mean)

            ## Episode Reward
            if self.episode_delta > self.sum_mean:
                print("Episode reward is 1000")
                self.reward = self.reward + 1000
                self.episode_delta = self.sum_mean
            else:
                print("Episode reward is -1000")
                self.reward = self.reward - 1000
                self.episode_delta = self.sum_mean
            self.sum_mean = 0
            self.word = ''
            self.delta = 0
            self.l = 0
            score = 0
            return self.reward, done, score

        self.image = cv2.putText(self.image, self.word, (70, 260), font, fontScale, color, thickness, cv2.LINE_AA)
        ## Reward generation
        d1 = ManhattanMetric(vector1=self.target, vector2=self.image)
        arr = self.target - self.image
        mean = np.mean(arr) + 0.0000001 * d1
        self.sum_mean += mean
        print("The mean is :", mean)
        if self.n_games == 0:
            self.reward = 10
            self.delta = mean
            self.episode_delta = self.sum_mean
            self.l += 1

        else:
            if mean <= 0.15:
                self.reward = 10000
                done = False
                score = 0
                self.l += 1
                return self.reward, done, score
            elif mean < self.delta:
                self.reward = 100
                self.delta = mean
                done = False
                score = 0
                self.l += 1
                return self.reward, done , score

            else:
                self.reward = -100
                self.delta = mean
                done = False
                score = 0
                self.l += 1
                return self.reward, done, score
        done = False

        score = 0
        return self.reward, done, score


def train():
    agent = Agent()

    while True:
        #get old state
        state_old = agent.get_state()

        #get move
        final_move = agent.get_action(state_old)
        # perform move and get new state
        reward, done, score = agent.play_step(final_move)
        print("Reward is :", reward)


        state_new = agent.get_state()

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        print("short")
        if done:

            #train long memory, plot result

            agent.n_games += 1
            agent.train_long_memory()
            print("long")
            print("EPISODE ENDED : ", agent.n_games)
            print("*" * 1000)

    print("Saving the model")
    agent.model.save()


if __name__ == '__main__':
    train()