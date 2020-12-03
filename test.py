import numpy as np
from time import time
from random import random as rand

def movement_to_steps1(x,y,T):
    ''' Zwraca krok zamiast pozycji.'''
    s_x = [0] * (T - 1)
    s_y = [0] * (T - 1)
    for i in range(1, T):
        s_x[i-1] = x[i] - x[i-1]
        s_y[i-1] = y[i] - y[i-1]
    return s_x, s_y

def movement_to_steps2(x,y,T):
    ''' Zwraca krok zamiast pozycji.'''
    s_x = np.zeros(T - 1)
    s_y = np.zeros(T - 1)
    for i in range(1, T):
        s_x[i-1] = x[i] - x[i-1]
        s_y[i-1] = y[i] - y[i-1]
    return s_x, s_y

def movement_to_steps3(x,y,T):
    ''' Zwraca krok zamiast pozycji.'''
    s_x = []
    s_y = []
    for i in range(1, T):
        s_x.append(x[i] - x[i-1])
        s_y.append(y[i] - y[i-1])
    return s_x, s_y

print("1 z lista: ", end='')
start = time()
movement_to_steps1(x, y, 10000)
stop = time()
print(stop - start)
print("1 z array: ", end='')
start = time()
movement_to_steps1(xx, yy, 10000)
stop = time()
print(stop - start)
print("2 z lista: ", end='')
start = time()
movement_to_steps2(x, y, 10000)
stop = time()
print(stop - start)
print("2 z array: ", end='')
start = time()
movement_to_steps2(xx, yy, 10000)
stop = time()
print(stop - start)
print("3 z lista: ", end='')
start = time()
movement_to_steps3(x, y, 10000)
stop = time()
print(stop - start)
print("3 z array: ", end='')
start = time()
movement_to_steps1(xx, yy, 10000)
stop = time()
print(stop - start)