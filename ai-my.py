# Self-Driving Car 
#
# Created by Yarko

import numpy as np
import random
import torch
import torch.autograd



class NeuralNetwork(torch.nn.Module):
    """
    The architecture of our neural network.

    Parameters:
        n_states: The number of elements in the input state vector (3 signals from the sensors, orientation, and -orientation).
        n_actions: The number of possible actions to take.
    """
    
    def __init__(self, n_states, n_actions):
        
        super().__init__(self)
        
        #self.n_states = n_states
        #self.n_actions = n_actions

        # create the hidden layer of 32 nodes and the output layer to convert 
        # the states into actions
        self.fc1 = torch.nn.Linear(n_states, 32)    
        self.fc2 = torch.nn.Linear(32, n_actions)

    
    def forward(self, state):
        """
        Performs forward propagation, activating the neurons of the neural network.

        Parameters:
            state: A tensor containing 3 signals from the sensors + orientation + -orientation.

        Returns:
            Q-values for each possible actions according to the input state.
        """

        x = torch.nn.functional.relu(self.fc1(state))
        q = self.fc2(x)
        return q




class ReplayMemory():
    """
    Experience replay allows to learn from multiple different experiences from the past,
    because learning from one experience at a time is not sufficient. The memory window
    stores previous experiences (transitions), including rare cases like sharp turns,
    so they don't get forgotten. This memory will be used to predict future actions.

    Parameters:
        capacity: The size of the memory window.
    """

    def __init__(self, capacity):
        
        # the number of transitions to store
        self.capacity = capacity

        # the list of previous transitions
        self.transitions = []


    
    def push(self, transition):
        """
        Appends a new transition to the memory. The size of the memory is limited by the capacity parameter.
        
        Parameters:
            transition: A tuple of tensors representing the last state, the new state, the last action, and the last reward.
        """
        
        # transition = (last state, new state, last action, last reward)
        self.transitions.append(transition)
        if len(self.transitions)>self.capacity:
            del self.transitions[0]

    
    def sample(self, batch_size):
        """
        Extracts random samples from the memory. Randomization prevents from biasing to similar past experiences,
        e.g. driving along the straight road. Turn experiences still have a chance to be selected from the memory.

        Parameters:
            batch_size: The number of samples to extract.

        Returns:
            An iterator to a sequence of tensors representing batches of last states, 
            new states, last actions, and last rewards.
        """

        # get batch_size random transitions
        samples = random.sample(self.transitions, batch_size)
        # zip will group states, new states, actions, and rewards into batches (each batch is a tuple of batch_size)
        samples = zip(*samples)

        # torch.cat() will convert each tuple into a tensor with elements in rows (0 dimension)
        # requires_grad_() enables gradient calculation in place
        samples = map(lambda x: torch.cat(x, 0).requires_grad_(True), samples)
        return samples

test_mem = ReplayMemory(10)
test_mem.push((torch.FloatTensor([1.1]), torch.FloatTensor([2.2]), torch.FloatTensor([3.3]), torch.FloatTensor([4.4])))
test_mem.push((torch.FloatTensor([9.9]), torch.FloatTensor([8.8]), torch.FloatTensor([7.7]), torch.FloatTensor([6.6])))
test_mem.push((torch.FloatTensor([10.0]), torch.FloatTensor([11.1]), torch.FloatTensor([12.2]), torch.FloatTensor([13.3])))
samples = test_mem.sample(3)