# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 18:04:39 2022

@author: ozano
"""

import numpy as np
import pandas as pd
global seed_number
seed_number=0

class Network:
    def sigmoid(self, x):
        return 1/(1+np.exp(-1*x))
    def get_derivative(self, x):
        return np.dot(self.sigmoid(x),(1-self.sigmoid(x)))
    def set_dataset(self, dataset):
        self.train=np.array(dataset)
    def __init__(self, train, output):
        np.random.seed(100)
        self.epsilon=0.12
        self.train=np.insert(train,0,np.ones(len(train)),axis=1)
        self.output=np.array(output)
        self.theta1=np.random.rand(len(self.train[0]))*self.epsilon-self.epsilon
        self.lambda_val=0.1
        self.alpha=0.2
        self.theta2=np.random.rand(2)*self.epsilon-self.epsilon
        np.random.seed(0)
    def train_network(self):
        for a in range(650):
            for x in range(len(self.train)):
                a1=self.train[x]
                z2=np.matmul(a1,self.theta1.reshape(len(self.theta1),1))
                a2=np.insert(self.sigmoid(z2), 0, 1, axis=0)
                z3=np.matmul(a2,self.theta2.reshape(len(self.theta2),1))
                a3=self.sigmoid(z3)
                sigma3=a3-self.output[x]
                sigma2=np.dot(sigma3*self.theta2,self.get_derivative(np.insert(z2,0, 1,axis=0)))
                sigma2=sigma2[1:]
                delta1=np.dot(sigma2[0], a1)
                delta2=np.dot(sigma3[0], a2)
                p1=self.lambda_val/len(self.train)*np.insert(self.theta1[1:],0, 0, axis=0)
                p2=self.lambda_val/len(self.train)*np.insert(self.theta2[1:],0, 0, axis=0)
                self.theta1=self.theta1-self.alpha*delta1/len(self.train)+p1
                self.theta2=self.theta2-self.alpha*delta2/len(self.train)+p2
    def predict(self, X):
        a1=np.insert(X, 0, 1, axis=0)
        z2=np.matmul(a1, self.theta1.reshape(len(self.theta1),1))
        a2=np.insert(self.sigmoid(z2),0, 1, axis=0)
        z3=np.matmul(a2, self.theta2.reshape(len(self.theta2), 1))
        return self.sigmoid(z3)
class GamePlayer:
    def __init__(self, n_n):
        self.n_n=n_n
        self.table=np.zeros([3,3])
        print(self.table)
        self.turn_x=True
    def play(self):
        game_cont=True
        while game_cont:
            if self.turn_x:
                print("Enter your move: ")
                move=input()
                move=int(move)
                self.table[int(move/3)][move%3]=-1
                if self.check_if_game_over(move):
                    game_cont=False
                self.turn_x=False
            else:
                (move_y, move_x)=self.think()
                self.table[move_y][move_x]=1
                if self.check_if_game_over(move_y*3+move_x):
                    game_cont=False
                self.turn_x=True
            print(self.table)
    def check_if_game_over(self, move):
        if self.turn_x:
            y=int(move/3)
            x=move%3 
            if x-2>=0 and y-2>=0:
                if self.table[y][x]+self.table[y-1][x-1]+self.table[y-2][x-2]==-3:
                    return True
            elif x+2<3 and y+2<3:
                if self.table[y][x]+self.table[y+1][x+1]+self.table[y+2][x+2]==-3:
                    return True
            elif sum(self.table[:][x])==-3:
                return True
            elif sum(self.table[y][:])==-3:
                return True
            else:
                self.turn_x=False
                return False
        else:
            y=int(move/3)
            x=move%3 
            if x-2>=0 and y-2>=0:
                if self.table[y][x]+self.table[y-1][x-1]+self.table[y-2][x-2]==3:
                    return True
            elif x+2<3 and y+2<3:
                if self.table[y][x]+self.table[y+1][x+1]+self.table[y+2][x+2]==3:
                    return True
            elif sum(self.table[:][x])==3:
                return True
            elif sum(self.table[y][:])==3:
                return True
            else:
                self.turn_x=True
                return False
    def think(self):
        moves_x=[]
        moves_y=[]
        percentages=[]
        for x in range(len(self.table)):
            for y in range(len(self.table[0])):
                copy_table=np.array(self.table)
                if self.table[x][y]==0:
                    copy_table[x][y]=1
                    val=self.n_n.predict(copy_table.reshape(9))
                    moves_x.append(y)
                    moves_y.append(x)
                    percentages.append(val)
        max_val=percentages[0]
        max_x=moves_x[0]
        max_y=moves_y[0]
        for a in range(len(percentages)-1):
            if percentages[a+1]>max_val:
                max_x=moves_x[a+1]
                max_y=moves_y[a+1]
                max_val=percentages[a+1]
        return (max_y, max_x)
data=pd.read_csv('tic-tac-toe-endgame.csv')
data=data.to_numpy()
print(data)
for y in range(len(data)):
    for x in range(len(data[y])):
        if data[y][x]=='x':
            data[y][x]=-1
        elif data[y][x]=='o':
            data[y][x]=1
        elif data[y][x]=='b':
            data[y][x]=0
        elif data[y][x]=='positive':
            data[y][x]=0
        elif data[y][x]=='negative':
            data[y][x]=1
print(data)
games=np.delete(data, len(data[0])-1, axis=1).astype(float)
outputs=np.array([x[len(data[0])-1] for x in data]).astype(float)
n_n=Network(games, outputs)
n_n.train_network()
player=GamePlayer(n_n)
player.play()

                        