# Self-Driving Car 
#
# Created by Yarko

import numpy as np
import random
import torch
#import torch.autograd



class NeuralNetwork(torch.nn.Module):
    """
    The architecture of our neural network.

    Parameters:
        n_states: The number of elements in the input state vector (3 signals from the sensors, orientation, and -orientation).
        n_actions: The number of possible actions to take.
    """
    
    def __init__(self, n_states, n_actions):
        
        super().__init__()
        
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

    def size(self):
        """        
        Returns:
            The actual number of entries in the memory.
        """
        return len(self.transitions)
    
    '''
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

        # torch.cat() will convert each batch (tuple) into a tensor with elements in rows (0 dimension)
        # requires_grad_() enables gradient calculation in place
        samples = map(lambda x: torch.cat(x, 0).requires_grad_(True), samples)
        return samples
    '''

    def sample(self, batch_size):
        """
        Extracts random samples from the memory. Randomization prevents from biasing to similar past experiences,
        e.g. driving along the straight road. Turn experiences still have a chance to be selected from the memory.

        Parameters:
            batch_size: The number of samples to extract.

        Returns:
            An iterator to a sequence of batches of last states, new states, last actions, and last rewards.
        """

        samples = random.sample(self.transitions, batch_size)
        samples = zip(*samples)
        # do we really need gradients for our samples?
        #samples = map(lambda x: torch.tensor(x, requires_grad=True), samples)        
        samples = map(lambda x: torch.tensor(x), samples)
        return samples

""" test_mem = ReplayMemory(10)
test_mem.push((1.1, 2.2, 3, 4.4))
test_mem.push((9.9, 8.8, 7, 6.6))
samples = test_mem.sample(2)
b1, b2, b3, b4 = samples
 """
""" test_mem = ReplayMemory(10)
test_mem.push((torch.FloatTensor([1.1]), torch.FloatTensor([2.2]), torch.FloatTensor([3.3]), torch.FloatTensor([4.4])))
test_mem.push((torch.FloatTensor([9.9]), torch.FloatTensor([8.8]), torch.FloatTensor([7.7]), torch.FloatTensor([6.6])))
test_mem.push((torch.FloatTensor([10.0]), torch.FloatTensor([11.1]), torch.FloatTensor([12.2]), torch.FloatTensor([13.3])))
samples = test_mem.sample(3)
b1, b2, b3, b4 = samples
print(b1, b2, b3, b4) """


class Brain():
    """
    Deep Q-Learning implementation.

    Parameters:
        n_states: The number of elements in the input state vector (3 signals from the sensors, orientation, and -orientation).
        n_actions: The number of possible actions the car can make.
        gamma: The discount factor in the Bellman equation Q(s,a) = R(s,a) + gamma*max(a': Q(s',a')).
            The closer it is to 0, the more AI will try to optimize the current reward. If it is closer to 1,
            AI will try to optimize the future reward more. 
    """

    def __init__(self, n_states, n_actions, gamma):
        self.gamma = gamma        
        self.model = NeuralNetwork(n_states, n_actions)
        self.memory = ReplayMemory(100000)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.last_state = (0,)*n_states 
        self.last_action = 0
        self.last_reward = 0
        # a history of rewards that will be used for calculation of the mean reward over last N rewards        
        self.reward_window = []


    def choose_action(self, state, temperature=10):
        """
        Randomly draw an action according to the probability distribution of possible actions calculated via softmax. 
        The AI A-Z Handbook suggests that it's a better method than selecting an action with the highest Q(s,a).

        Parameters:
            state: The current state.
            temperature: Controls the confidence of the choice. The higher the temperature, the more likely actions
                with high probabilities will be chosen. 

        Returns:
            The action selected.
        """

        with torch.no_grad(): # we don't need gradient computation here
            t = torch.tensor(state, dtype=torch.float)
            out = self.model(t)
            p = torch.nn.functional.softmax(out * temperature, dim=-1)

        # In probability theory, the multinomial distribution is a generalization of the binomial distribution. 
        # For example, it models the probability of counts for rolling a k-sided die n times. For n independent 
        # trials each of which leads to a success for exactly one of k categories, with each category having a 
        # given fixed success probability, the multinomial distribution gives the probability of any particular 
        # combination of numbers of successes for the various categories.
        # When k is 2 and n is 1, the multinomial distribution is the Bernoulli distribution. When k is 2 and 
        # n is bigger than 1, it is the binomial distribution. When k is bigger than 2 and n is 1, it is the 
        # categorical distribution.
        action = p.multinomial(num_samples=1)
        return action.item()

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        """
        Adjust the weights of the neural network to better fit the environment.

        Parameters:
            batch_state: A tuple of states from the memory.
            batch_next_states: A tuple of states following the states from batch_state.
            batch_action: A tuple of actions taken at each state from batch_state.
            batch_reward: A tuple of reward values received after taking actions from batch_action being in 
                the states from batch_state.
        """

        # get the list of Q-values for each state sampled from the memory
        q = self.model(batch_state)
        
        # from the list of Q-values for each state in the batch_state we need only those which have been chosen
        q = q.gather(1, batch_action.unsqueeze(1))

        # get Q-values for the next states
        q_next = self.model(batch_next_state)

        # Get the best Q-value for each next state from the batch.
        # To use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().
        # Probably, we could disable gradients after calling max().
        # https://discuss.pytorch.org/t/detach-no-grad-and-requires-grad/16915/6
        q_next, _ = q_next.detach().max(dim=1)  

        # the target Q-values are computed with the Bellman equation: R(s,a) + gamma*max(a': Q(s',a'))
        q_target = batch_reward + self.gamma * q_next

        # Creates a criterion that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise. 
        # It is less sensitive to outliers than the MSELoss and in some cases prevents exploding gradients. 
        # Also known as the Huber loss:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
        loss = torch.nn.functional.smooth_l1_loss(q, q_target)

        # We need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients 
        # on subsequent backward passes.
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        # https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        self.optimizer.zero_grad() 

        # Setting retain_graph = True is claimed to improve performance:
        # https://www.udemy.com/artificial-intelligence-az/learn/lecture/7147758#questions
        # However, from PyTorch documentation:
        # If False, the graph used to compute the grads will be freed. Note that in nearly all cases setting this option to True 
        # is not needed and often can be worked around in a much more efficient way.
        loss.backward(retain_graph = True)

        # adjust the weights of the neural network
        self.optimizer.step()


    def update(self, reward, new_state):
        """
        Adds a new transition to the memory and adjusts the neural network.

        Parameters:
            reward: The reward for taking the action which led to the new_state.
            new_state: The state after taking the last action. 

        Returns:
            A new action to take.
        """

        self.memory.push((self.last_state, new_state, self.last_action, self.last_reward))

        self.last_state = new_state
        self.last_action = self.choose_action(new_state, 100)
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        if self.memory.size() >= 2:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(2)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)

        return self.last_action



    def save(self):
        pass

    def load(self):
        pass


brain = Brain(5, 3, 0.3)
a = brain.update(1., (1.,2.,3.,4.,5.))
a = brain.update(2, (0.1, 0.2, 0.3, 0.4, 0.5))
a = brain.update(0, (1.6, 1.7, 1.8, 1.9, 1.1))

#a = brain.choose_action((1,2,3,4,5))
#brain.update((1.,2.,3.,4.,5.), (6.,7.,8.,9.,10.), 1, 2.)
#brain.update((0.1,0.2,0.3,0.4,0.5), (0.6,0.7,0.8,0.9,0.1), 0, 4.)
#brain.update((1.1,1.2,1.3,1.4,1.5), (1.6,1.7,1.8,1.9,1.1), 2, 3.)
print(a)