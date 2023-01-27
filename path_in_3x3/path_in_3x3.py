#!/home/akoel/Projects/python/q-learning/path_in_3x3/env/bin/python

import numpy as np

#
# Game Board 3x3
#
# 00 01 02
# 10 11 10
# 20 21 22
#

states=["00", "01", "02", "10", "11", "12", "20", "21", "22"]

# 0 1 2 3 4
# → ↑ ← ↓ idle
actions=[0,1,2,3,4]

rewards=[
    [("01",1), (None,0), (None,0), ("10",1), ("00",0)],
    [("02",1), (None,0), ("00",1), ("11",1), ("01",0)],
    [(None,0), (None,0), ("01",1), ("12",1), ("02",0)],
    [("11",1), ("00",1), (None,0), ("20",1), ("10",0)],
    [("12",1), ("01",1), ("10",1), ("21",1), ("11",0)],
    [(None,0), ("02",1), ("11",1), ("22",1), ("12",0)],
    [("21",1), ("10",1), (None,0), (None,0), ("20",0)],
    [("22",1), ("11",1), ("20",1), (None,0), ("21",0)],
    [(None,0), ("12",1), ("21",1), (None,0), ("22",0)]
]

# Initialize parameters
gamma = 0.75 # Discount factor 
alpha = 0.9 # Learning rate 

endState = "22"
stateNumbers=len(states)
actionNumbers=len(actions)

def getOptimalRoute(startState, endState):
    endStateIndex = states.index(endState)
    rewards[endStateIndex][actionNumbers-1] = ("22",999)

    rewards[1][3] = ("11", -800)
    rewards[3][0] = ("11", -800)
    rewards[5][2] = ("11", -800)
    rewards[7][1] = ("11", -800)



    # Initializing Q-Values
    Q = np.array(np.zeros([stateNumbers, actionNumbers]))

    # Q-Learning process
    for i in range(1000):

        # Pick up a state randomly
        currentStateIndex = np.random.randint(0, stateNumbers) # Python excludes the upper bound

        # For traversing through the neighbors
        possibleActions = []

        # Iterate through the new rewards matrix and get the actions > 0
        for j in range(actionNumbers):

            # considers only positive rewards
            nextAndRew = rewards[currentStateIndex][j]
            if nextAndRew[1] > 0 and nextAndRew[0] is not None:
                possibleActions.append(j)

        # Pick an action randomly from the list of posible actions
        nextActionIndex = np.random.choice(possibleActions)
        nextState = rewards[currentStateIndex][nextActionIndex][0]
        nextStateIndex = states.index(nextState)

        reward = rewards[currentStateIndex][nextActionIndex][1]
        TD = reward + gamma * Q[nextStateIndex, np.argmax(Q[nextStateIndex,])] - Q[currentStateIndex, nextActionIndex]

        # Update the Q-Value using the Bellman equation
        Q[currentStateIndex,nextActionIndex] += alpha * TD

#        print("{0}, {1} {2}->{3} {4}".format(states[currentStateIndex], possibleActions, nextActionIndex, nextState, rewards[currentStateIndex][nextActionIndex]))

        # Initialize the optimal route with the starting location
        route = [startState]

        # We do not know about the next location yet, so initialize with the value of 
        # starting location
        nextState = startState

        # We don't know about the exact number of iterations
        # needed to reach to the final location hence while loop will be a good choice 
        # for iteratiing

    while(nextState != endState):

        actualStateIndex = states.index(nextState)

        actionIndex=np.argmax(Q[actualStateIndex,])

        nextState=rewards[actualStateIndex][actionIndex][0]

#        print("{0} {1} {2}".format(actualStateIndex, actionIndex, nextState))

        route.append(nextState)

#    print(Q)
    return route


route=getOptimalRoute("00", "22")
print(route)




















