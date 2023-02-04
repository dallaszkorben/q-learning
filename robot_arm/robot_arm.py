#!/home/akoel/Projects/python/ai/q-learning/robot_arm/env/bin/python

import time
import numpy as np
import math 
import pickle
from functools import partial
from pprint import pprint
from robot_arm_animate import RobotArmAnimate 
from shapely.geometry import LineString
from shapely.geometry import Polygon

class RobotArm(object):

    GOAL_REWARD = 999

    def __init__(self, angle_step, alpha=0.9, gamma=0.75, goal_position=((95,5),(105,15)), blocker_position_list=[]):

        self.alpha = alpha			# Learning rate
        self.gamma = gamma			# Discount factor 
        self.angle_step = angle_step
        self.goal_position = goal_position
        self.blocker_position_list = blocker_position_list
        self.blocker_rectangles_list = []

        self.Q = {}

        # Arms
        self.arms_list = [
            {"name": "arm0", "length": 100, "min_angle": -45, "max_angle": 45},
            {"name": "arm1", "length": 80, "min_angle": -45, "max_angle": 135},
            {"name": "arm2", "length": 50, "min_angle": -45, "max_angle": 90}
        ]

        self.actions=[
            partial(self.get_state_with_increased_angle, increase_by=angle_step, arm_index=0),
            partial(self.get_state_with_decreased_angle, decrease_by=angle_step, arm_index=0),
            partial(self.get_state_with_increased_angle, increase_by=angle_step, arm_index=1),
            partial(self.get_state_with_decreased_angle, decrease_by=angle_step, arm_index=1),
            partial(self.get_state_with_increased_angle, increase_by=angle_step, arm_index=2),
            partial(self.get_state_with_decreased_angle, decrease_by=angle_step, arm_index=2),
            partial(self.get_state_with_idle),
        ]


    def print_progress_bar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', print_end = "\r"):
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
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)

        # Print New Line on Complete
        if iteration == total: 
            print()

    def get_degree(self, state, index=None):
        """
        It pharses and gives back the degree out of State
        If the index parameter is not provided, then it gives back all arm's angle in a list as number (int)
        If the index is provided, then it gives back the angle of the index.th arm

        :param state: The hash of state, representing the arm's angle in degrees. Like "20_-10_45"
        :param index: The index of the arm which angle's we are interested in
        :type state: string
        :type index: int
        :return: Angle of the arm(s)
        :rtype: int/list
        """

        if index == None:
            return [int(i) for i in state.split("_")]
        return int(state.split("_")[index])

    def get_arms_end_positions(self, state):
        """
        gives back the positions (x,y) of the end-points of all arms in a list in the given state
        :param state: The hash of state, representing the arm's angle in degrees. Like "20_-14_45"
        :type state: string
        :return: Positions. [(x0,y0,fi0),(x1,y1,fi1),(x2,y2,fi2)]
        :rtype: list
        """

        degree_list = self.get_degree(state)
        prev_pos = (0,0)
        prev_angle = 0
        pos = []
        for index, arm in enumerate(self.arms_list):
            angle = np.pi*(degree_list[index])/180 + prev_angle
            prev_angle = angle
            y = arm["length"] * math.cos(angle) + prev_pos[1]
            x = arm["length"] * math.sin(angle) + prev_pos[0]
            prev_pos = (x, y, degree_list[index])
            pos.append(prev_pos)
        return pos

    def get_state_with_increased_angle(self, increase_by, arm_index, actual_state):
        """
        Calculates and returns back the new state after the given arm's angle was increased by the given degree regardless of the new angle is allowed
        """

        separate_list = actual_state.split("_")
        separate_list[arm_index] = str(int(separate_list[arm_index]) + increase_by)
        return "_".join(separate_list), arm_index

    def get_state_with_decreased_angle(self, decrease_by, arm_index, actual_state):
        """
        Calculates and returns back the new state after the given arm's angle was decreased by the given degree regardless of the new angle is allowed
        """

        separate_list = actual_state.split("_")
        separate_list[arm_index] = str(int(separate_list[arm_index]) - decrease_by)
        return "_".join(separate_list), arm_index

    def get_state_with_idle(self, actual_state):
        """
        Returns back the given State as the new State representing Idle action
        """

        return actual_state, None

    def is_zero(self, value):
        """
        Checks if a value is not exaclty zero but very close to it.
        This is needed as the conversion of degree to rad is not very accurate, so the cos(90) will not give back exactly 0 (same with sin(0))
        """

        return np.isclose(value, 0, rtol=1e-05, atol=1e-08, equal_nan=False)

    def is_valid_state(self, actual_state):
        """
        It checkes if the actual State is valid.
        It is valid, if
        - all joints are above the ground: y > 0
        - all the angles of the arms are between the range configured in the "self.arms_list"
        """

        pos=self.get_arms_end_positions(actual_state)

        return self.is_valid_position(pos)

    def is_valid_position(self, actual_position):
        """
        It checkes if the actual position is valid.
        It is valid, if
        - all joints are above the ground: y > 0
        - all the angles of the arms are between the range configured in the "self.arms_list"
        """

        all_ok = True
        arm_start_pos = (0, 0)
        # go through all arms
        for index, arm in enumerate(self.arms_list):

            arm_end_pos = (actual_position[index][0], actual_position[index][1])

            # one joint is below the ground: y < 0
            if arm_end_pos[1] < 0 and not self.is_zero(arm_end_pos[1]):
                all_ok = False
                return False
                #break

            # one angle is out of the valid range
            if actual_position[index][2] > arm["max_angle"] or actual_position[index][2] < arm["min_angle"]:
                all_ok = False
                return False
                #break

            # Blocked position
            line = LineString([arm_start_pos, arm_end_pos])
            for rectangle in self.blocker_rectangles_list:
                if line.crosses(rectangle):
                    return False

            arm_start_pos = arm_end_pos
        return all_ok

    def is_goal_position(self, pos):
        arm_index = len(self.arms_list) - 1

        # x, y reached the end-position (goal)
        if pos[arm_index][0] >= self.goal_position[0][0] and pos[arm_index][0] <= self.goal_position[1][0] and pos[arm_index][1] >= self.goal_position[0][1] and pos[arm_index][1] <= self.goal_position[1][1]:
            return True
        return False

    def is_prefered_state(self, state):
        if state == "0_0_0":
            return True
        return False

    def get_possible_states_with_reward(self, actual_state):
        """
        Gives back the possible state-reward tuple list in the given state
        :param state: The hash of the actual State, representing the arm's angle in degrees. Like "20_-14_45"
        :type state: string
        :return: list of possible next (State-Reward) tuples
        :rtype: list of tuples
        """

        states = []

        # go through the Actions
        for index, func in enumerate(self.actions):

            # get the new State in case of the next Action
            new_state, arm_index = func(actual_state=actual_state)

            # The default Reward for the Action is 0
            reward = 0

            # Fetches the list of arm's positions to be able to calculate if 
            # -the Action is valid (reward=1) 
            # -the Action is in the end-position (reward=999)
            # -the Action is in a Special position which we what to reach anyway in the sequence to go through from the start-position to the end-position (reward=100)
            pos=self.get_arms_end_positions(new_state)

            # all joints are above the ground and all angles in the valid range
            if self.is_valid_position(pos):

                # The Action is IDLE: No arm was moved
                if arm_index is None:

                    # the arm reached the goal position
                    if self.is_goal_position(pos):

                        reward = self.GOAL_REWARD
                        self.is_goal_reward = True

#                elif self.is_prefered_state(new_state): 
#                    reward = 10
                else:
                    reward = 1

                states.append({"new_state":new_state, "reward":reward, "action_index":index})
        return states


    def recursively_fill_up_q_table(self, arm_index, preposition_list):
        """
        This recursive method fills up the Q table.
        It builds as many nested for-loop as many arm we have.
        Every loop goes through all the theoretically possible angles. '
        In the deepest loop it will check if there is at least one possible Action in the actual State
        If there is, then this State is added to the Q table with 0 values
        If there is No, then this State will be ignored
        """
        for i in range(self.arms_list[arm_index]["min_angle"], self.arms_list[arm_index]["max_angle"] + self.angle_step, self.angle_step):
            actual_preposition_list = preposition_list + [str(i)]

            if arm_index < len(self.arms_list) - 1:
                self.recursively_fill_up_q_table(arm_index + 1, actual_preposition_list)

            # the deepest loop
            else:
                state = "_".join(actual_preposition_list)

                self.bar_index += 1
                self.print_progress_bar(self.bar_index, self.bar_size, prefix = 'Fill up progress:', suffix = 'Complete', length = 50)

                # Should we have to filter out the States which have NO valid next State or not?

                if self.is_valid_state(state):
                    vector = [0,0,0,0,0,0,0]
                    self.Q[state] = vector

    def fill_up_q_table(self):
        """
        Generates a new Q table with 0 values in all States for all Actions.
        It considers only the States which have at least one posible Action.
        """

        # calculate the theoretical size of the States
        self.bar_size = 1
        for arm in self.arms_list:
            self.bar_size *= (arm["max_angle"] - arm["min_angle"] + self.angle_step) / self.angle_step
        self.bar_index = 0

        self.Q = {}

        # indicates that there is NO goal reward found for the Q
        self.is_goal_reward = False

        self.recursively_fill_up_q_table(0, [])

        print("Length of State list: {0}".format(len(self.Q)))

    def training(self, start_training_steps=0, end_training_steps=100000, file_name="my_file"):
        """
        Trains the robot arm in the given train_steps times.
        It takes the Q table, picks up randomly a State and a possible next State and calculates the Q value for the Action
        """

        # generates the blocler_rectangles_list
        self.blocker_rectangles_list = []
        for blocker_pos in blocker_position_list:
            rectangle = Polygon([blocker_pos[0], [blocker_pos[1][0], blocker_pos[0][1]], blocker_pos[1], [blocker_pos[0][0],blocker_pos[1][1]]])
            self.blocker_rectangles_list.append(rectangle)

        # empty Q table
        if not self.Q:

            # generates and fills up the Q table
            self.fill_up_q_table()

        # generates the possible state list out of the Q table
        state_list = list(self.Q.keys())

        print("{0} - {1}".format(start_training_steps, end_training_steps))

        for t in range(start_training_steps, end_training_steps):

            # progress bar
            ##self.print_progress_bar(t, training_steps-1, prefix = 'Train   Progress:', suffix = 'Complete with goal reward ' if self.is_goal_reward else 'Complete ', length = 50)
            self.print_progress_bar(t, end_training_steps-1, prefix = 'Train   Progress:', suffix = "Complete ({0}/{1}) - {2}".format(end_training_steps, t, "goal reward found" if self.is_goal_reward else ""),length = 50)

            # pick up a random state
            from_state = np.random.choice(state_list)

            # collects the possibel states from the actual state
            possible_states_with_reward_list = self.get_possible_states_with_reward(from_state)

            # picks up a random state out of the possible states IF the Action's Reward is > 0. The Reward=0 means, the State is not reachable from the actual State
            #print(possible_states_with_reward_list)


            # The dilema here is that we have states which have NO valid next state
            # To filter them out in the fill_up_q_table() method takes a lot of time
            # So temporarialy we keep it as it is
            if any(possible_state['reward'] > 0 for possible_state in possible_states_with_reward_list):
                to_state_with_reward = np.random.choice([item for item in possible_states_with_reward_list if item["reward"] > 0])

                # collects data for the calculation
                to_state = to_state_with_reward["new_state"]
                action_index = to_state_with_reward["action_index"]
                action_reward = to_state_with_reward["reward"]

                Q_actual = self.Q[from_state][action_index]
                Q_next_max = np.max(self.Q[to_state])

                # calculates the Bellman equation
                TD = action_reward + self.gamma * Q_next_max - Q_actual
                self.Q[from_state][action_index] += self.alpha * TD

            # save Q table after every 10000 steps
            if t % 10000 == 0:
                my_robot_arm.save_q(file_name, t)

        my_robot_arm.save_q(file_name, end_training_steps)

        if self.is_goal_reward:
            return True

        else:
            print("!!! There was NO goal Action found !!!")
            return False










    def systematic_training(self, start_training_steps=0, end_training_steps=100, file_name="my_file"):
        """
        Trains the robot arm in the given train_steps times.
        It takes the Q table, picks up randomly a State and a possible next State and calculates the Q value for the Action
        """

        # generates the blocler_rectangles_list
        self.blocker_rectangles_list = []
        for blocker_pos in blocker_position_list:
            rectangle = Polygon([blocker_pos[0], [blocker_pos[1][0], blocker_pos[0][1]], blocker_pos[1], [blocker_pos[0][0],blocker_pos[1][1]]])
            self.blocker_rectangles_list.append(rectangle)

        # empty Q table
        if not self.Q:

            # generates and fills up the Q table
            self.fill_up_q_table()

        # generates the possible state list out of the Q table
        state_list = list(self.Q.keys())

        # records start time
        start_time = time.perf_counter()
        end_time = "?"
        elapsed_time = "?"

        for t in range(start_training_steps, end_training_steps):

            # progress bar
            ##self.print_progress_bar(t, training_steps-1, prefix = 'Train   Progress:', suffix = 'Complete with goal reward ' if self.is_goal_reward else 'Complete ', length = 50)
            self.print_progress_bar(t, end_training_steps-1, prefix = 'Train   Progress:', suffix = "Complete ({0}/{1}) ({2}s/{3}s) - {4}".format(end_training_steps, t, end_time, elapsed_time, "goal reward found" if self.is_goal_reward else ""),length = 50)

            # go through all states
            for from_state in state_list:

                # collects the possibel states from the actual state
                possible_states_with_reward_list = self.get_possible_states_with_reward(from_state)

                # go through all possible next states
                for to_state_with_reward in [item for item in possible_states_with_reward_list if item["reward"] > 0]:

                    # collects data for the calculation
                    to_state = to_state_with_reward["new_state"]
                    action_index = to_state_with_reward["action_index"]
                    action_reward = to_state_with_reward["reward"]

                    Q_actual = self.Q[from_state][action_index]
                    Q_next_max = np.max(self.Q[to_state])

                    # calculates the Bellman equation
                    TD = action_reward + self.gamma * Q_next_max - Q_actual
                    self.Q[from_state][action_index] += self.alpha * TD

            # save Q table after every 20 steps
            if t % 20 == 0:
                my_robot_arm.save_q(file_name, t)

            recent_time = time.perf_counter()
            elapsed_time = int(recent_time - start_time)

            elapsed_perc = (t+1) / float(end_training_steps)
            end_time = int(elapsed_time / elapsed_perc)

        my_robot_arm.save_q(file_name, end_training_steps)

        if self.is_goal_reward:
            return True

        else:
            print("!!! There was NO goal Action found !!!")
            return False




    def save_q(self, name, trained_steps):
        var_list = [self.alpha, self.gamma, self.angle_step, self.goal_position, self.blocker_position_list, self.arms_list, self.Q, trained_steps]
        with open(name, 'wb') as f: 
            pickle.dump(var_list, f)

    def load_q(self, name):
        with open(name, 'rb') as f:
            self.alpha, self.gamma, self.angle_step, self.goal_position, self.blocker_position_list, self.arms_list, self.Q, trained_steps = pickle.load(f)
        self.is_goal_reward = True
        return trained_steps

    def get_optimal_route(self, start_state):
        """

        """

        route = []
        next_state = start_state

        if not self.is_goal_reward:
            route.append(next_state)

        while next_state not in route:
            route.append(next_state)

            actual_state = next_state

            # if the state does NOT exist in the dict
            if not self.Q.get(actual_state):
                print("The Start-State is not valid")
                break

            reward_list = self.Q[actual_state]

            # it selects the first occurence of the highest value
            #next_action_index = np.argmax(reward_list)

            # this section responsible for picking up the Action with the highest Q value
            # If there are more than 1 highest value then it selects the first which is NOT in the route list yet
            # (the original strategy, to pick up the first Action was WRONG. Many times the alternative Actions with the same highest value, refer back to a State which was already selected)
            possible_next_action_index_list = np.argwhere(reward_list==np.amax(reward_list)).flatten().tolist()
            previous_arm_index = None
            i = 0

#            print("reward: {0}possible_next_action_index_list)

            while True:
                next_action_func = self.actions[possible_next_action_index_list[i]]
                next_state, arm_index = next_action_func(actual_state=actual_state)

                i += 1
                if next_state not in route or i >= len(possible_next_action_index_list):
                    break

        print(route)

        if route and start_state in route:
            my_robot_arm_animate = RobotArmAnimate()
            my_robot_arm_animate.animate(self.arms_list, route, self.goal_position, self.blocker_position_list, pause_between_states=0.2)

        return route

# ############################
#
# TODO
#
# -Alternative way to random
#
# ############################

training = False
to_continue = False

angle = 5
end_training_steps = 100
goal_position=((94,74),(96,76))
blocker_position_list=[((-110,0), (-90,250)), ((70, 76), (96, 250)), ((70, 0), (76, 50))]

my_robot_arm = RobotArm(angle_step=angle, goal_position=goal_position, blocker_position_list=blocker_position_list)

file_name = "{0}_{1}_{2}.q".format(angle, end_training_steps, str(goal_position).replace(" ",""))
start_training_steps = 0
if to_continue or not training:
    start_training_steps = my_robot_arm.load_q(file_name)
    print("Q-table loaded with {0} training steps".format(start_training_steps))
if training:
    print("Training starts from: {0} till: {1}".format(start_training_steps, end_training_steps))
#    my_robot_arm.training(start_training_steps, end_training_steps, file_name)
    my_robot_arm.systematic_training(start_training_steps, end_training_steps, file_name)
    print("Training finished ...")

#route = my_robot_arm.get_optimal_route('-15_100_-40')
route = my_robot_arm.get_optimal_route("0_0_0")


