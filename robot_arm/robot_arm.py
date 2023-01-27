#!/home/akoel/Projects/python/ai/q-learning/robot_arm/env/bin/python

import numpy as np
import math 
import pickle
from functools import partial
from pprint import pprint


angle_step = 45
gamma = 0.75 # Discount factor 
alpha = 0.9 # Learning rate 

# Arms
arms = [
{"name": "arm0", "length": 100, "min_angle": -90, "max_angle": 90},
{"name": "arm1", "length": 100, "min_angle": -90, "max_angle": 90},
{"name": "arm2", "length": 90, "min_angle": -90, "max_angle": 90}
]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_degree(status, index=None):
    """
    if the index parameter is not provided, then it gives back all arm's angle in a list as number (int)
    if the index is provided, then it gives back the angle of the index.th arm

    :param status: The hash of status, representing the arm's angle in degrees. Like "20_-10_45"
    :param index: The index of the arm which angle's we are interested in
    :type status: string
    :type index: int
    :return: Angle of the arm(s)
    :rtype: int/list
    """
    if index == None:
        return [int(i) for i in status.split("_")]
    return int(status.split("_")[index])

def get_position(status):
    """
    gives back the positions (x,y) of the end-points of all arms in a list in the given status
    :param status: The hash of status, representing the arm's angle in degrees. Like "20_-14_45"
    :type status: string
    :return: Positions. [(x0,y0,fi0),(x1,y1,fi1),(x2,y2,fi2)]
    :rtype: list
    """
    degree_list = get_degree(status)
    prev_pos = (0,0)
    prev_angle = 0
    pos = []
    for index, arm in enumerate(arms):
        angle = np.pi*(degree_list[index])/180 + prev_angle
        prev_angle = angle
        y = arm["length"] * math.cos(angle) + prev_pos[1]
        x = arm["length"] * math.sin(angle) + prev_pos[0]
        prev_pos = (x, y, degree_list[index])
        pos.append(prev_pos)
    return pos

def get_increased(increase_by, arm_index, actual_status):
    separate_list = actual_status.split("_")
    separate_list[arm_index] = str(int(separate_list[arm_index]) + increase_by)
    return "_".join(separate_list)

def get_decreased(decrease_by, arm_index, actual_status):
    separate_list = actual_status.split("_")
    separate_list[arm_index] = str(int(separate_list[arm_index]) - decrease_by)
    return "_".join(separate_list)

def get_idle(actual_status):
    return actual_status

def isZero(value):
    return np.isclose(value, 0, rtol=1e-05, atol=1e-08, equal_nan=False)

def get_possible_statuses_with_reward(actual_status):
    """
    Gives back the possible status-reward tuple list in the given status
    :param status: The hash of the actual status, representing the arm's angle in degrees. Like "20_-14_45"
    :type status: string
    :return: list of possible next (Status-Reward) tuples
    :rtype: list of tuples
    """


    """
    !!! 
    I should check if the actual_status is valid !!!
    I just check if the next status is valid
    !!!
    """
    statuses = []

    for index, func in enumerate(actions):

        new_status=func(actual_status=actual_status)

        reward = 0
        pos=get_position(new_status)

        # all joints are above the ground
        if pos[0][1] > 0 and pos[1][1] > 0 and pos[2][1] > 0 and not isZero(pos[0][1]) and not isZero(pos[1][1]) and not isZero(pos[2][1]):

            # all angles in the valid range
            if pos[0][2] <= 90 and pos[0][2] >= -90 and pos[1][2] <= 90 and pos[1][2] >= -90 and pos[2][2] <= 90 and pos[2][2] >= -90:

                # The Action is IDLE
                if new_status==actual_status:

                    # x, y reached the goal
                    if pos[2][0] >= 90 and pos[2][0] <= 105 and pos[2][1] >= 5 and pos[2][1] <= 15:
                        reward = 999

#                elif pos[2][0] >= 80 and pos[2][0] <= 1205 and pos[2][1] >= 0 and pos[2][1] <= 30:
##                    reward = 100
                else:
                    reward = 1

                statuses.append({"new_status":new_status, "reward":reward, "action_index":index})
    return statuses

actions=[
    partial(get_increased, increase_by=angle_step, arm_index=0),
    partial(get_decreased, decrease_by=angle_step, arm_index=0),
    partial(get_increased, increase_by=angle_step, arm_index=1),
    partial(get_decreased, decrease_by=angle_step, arm_index=1),
    partial(get_increased, increase_by=angle_step, arm_index=2),
    partial(get_decreased, decrease_by=angle_step, arm_index=2),
    partial(get_idle),
]

Q = {}
for i in range(-90, 100, angle_step):
    for j in range(-90, 100, angle_step):
        for k in range(-90, 100, angle_step):
            status = "_".join([str(i), str(j), str(k)])
            possible_status_list = get_possible_statuses_with_reward(status)

            # if it is valid status
            if possible_status_list:

                vector = [0,0,0,0,0,0,0]

                #for item in possible_status_list:
                #    vector[item["action_index"]] = item["reward"]

                Q[status] = vector






def train():
    status_list = list(Q.keys())
    train_steps = 100000
    for t in range(train_steps):
#    for from_status in ['0_90_45']: #, '90_0_90', '90_45_-45', '90_45_-90', '90_45_0', '90_45_45', '90_90_-45', '90_90_-90'] :

        printProgressBar(t, train_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)

        from_status = np.random.choice(status_list)

        possible_status_with_reward_list = get_possible_statuses_with_reward(from_status)

#        print(possible_status_with_reward_list)

        to_status_with_reward = np.random.choice([item for item in possible_status_with_reward_list if item["reward"] > 0])

#        print(to_status_with_reward)

        to_status = to_status_with_reward["new_status"]
        action_index = to_status_with_reward["action_index"]
        action_reward = to_status_with_reward["reward"]

        Q_actual = Q[from_status][action_index]
        Q_next_max = np.max(Q[to_status])

        TD = action_reward + gamma * Q_next_max - Q_actual
        Q[from_status][action_index] += alpha * TD

        #print(Q[to_status])
        #print("{0} => {1}: ({2}) Qmax:{3}".format(from_status, to_status, action_reward, Q_next_max))
        #print(possible_status_with_reward_list)

    saveQ()
#    pprint(Q)


def saveQ():
        name = "test"
        fw = open('q_' + str(name), 'wb')
        pickle.dump(Q, fw)
        fw.close()
        return fw

def loadQ():
        name = "test"
        fr = open('q_' + str(name), 'rb')
        Q = pickle.load(fr)
        fr.close()
        return Q
#        pprint(Q)


def get_optimal_route(start_status):
    route = []
    next_status = start_status

    while next_status not in route:
        route.append(next_status)

        actual_status = next_status
        reward_list = Q[actual_status]

#        print("{0}, {1}".format(actual_status, reward_list))


        next_action_index = np.argmax(reward_list)
        next_action_func = actions[next_action_index]
        next_status = next_action_func(actual_status=actual_status)
    print(route)



Q=loadQ()
get_optimal_route("-45_-45_-45")

#train()

#pos=get_possible_statuses_with_reward("-45_-45_-45")
#pos=get_position("0_90_90")
#print(pos)




