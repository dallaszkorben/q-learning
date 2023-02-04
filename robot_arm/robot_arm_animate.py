#!/home/akoel/Projects/python/ai/q-learning/robot_arm/env/bin/python

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pprint import pprint

class RobotArmAnimate(object):

    def __init__(self):
        self.quit = False
        self.stop = False
        self.animate_in_progress = False
        self.lines_to_delete_list = []

    def rotate_vector(self, data, angle):
        """
        Rotates the given points in the numpy data with the given angle
        0° is on the top, 180° is the bottom

        :param data: points to rotate in numpy array. array([[0, 0], [3, 8], ...])
        :type data: numpy array
        :param angle: the angle to rotate the points in degree
        :type angle: decimal
        :return: the rotated points
        :rtype: numpy array
        """
        # make rotation matrix
        theta = np.radians(angle)

        co = np.cos(theta)
        si = np.sin(theta)
        rotation_matrix = np.array(((co, -si), (si, co)))

        # rotate data vector
        rotated_vector = data.dot(rotation_matrix)

        # return index of elbow
        return rotated_vector

    def action_on_canvas(self, event):
        #print(event)

        if event.key:
            if event.key == "escape":
                self.quit = True
                plt.close()
            elif event.key == " ":
                if self.animate_in_progress:
                    self.stop = True
                else:
                    self.start_animate()
            #print("KEY click")
        elif event.button:
            pass
            #print("MOUSE click")
        else:
            pass

    def animate(self, arms_list, state_list, goal_square_points, blocker_position_list, pause_between_states=1.0):
        self.arms_list = arms_list
        self.state_list = state_list
        self.goal_square_points = goal_square_points
        self.blocker_position_list = blocker_position_list
        self.pause_between_states = pause_between_states

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.mpl_connect("key_press_event", self.action_on_canvas)
        self.fig.canvas.mpl_connect('button_press_event', self.action_on_canvas)

        plt.ylim(0, 300)
        plt.xlim(-100, 200)

        self.start_animate()

        plt.show()

    def start_animate(self):
        """
        """

        self.animate_in_progress = True
        self.stop = False

        # generates the end-points of the arms
        base_end_points_of_arms = [np.array([0, arm['length']]) for arm in self.arms_list]

        # Draws the Goal area
        #self.ax.add_patch(Rectangle((self.goal_square_points[0][0], self.goal_square_points[0][1]), self.goal_square_points[1][0]-self.goal_square_points[0][0], self.goal_square_points[1][1]-self.goal_square_points[0][1], facecolor='red', edgecolor='red', fill=True, linewidth=5))
        self.ax.plot(self.goal_square_points[0][0] + ((self.goal_square_points[1][0] - self.goal_square_points[0][0])/2), self.goal_square_points[0][1] + ((self.goal_square_points[1][1] - self.goal_square_points[0][1])/2), color='red', linewidth=5, markersize=10, marker='o')

        # Show the blockers
        for blocker_pos in self.blocker_position_list:
            self.ax.add_patch(Rectangle((blocker_pos[0][0], blocker_pos[0][1]), blocker_pos[1][0]-blocker_pos[0][0], blocker_pos[1][1]-blocker_pos[0][1], facecolor='gray', edgecolor='black', hatch='//', fill=True))

        for line in self.lines_to_delete_list:
            l=line.pop(0)
            l.remove()
        self.lines_to_delete_list = []

        # go through all the statuses to animate the move of arms
        for state in self.state_list:

            relative_angle_of_arms = [int(i) for i in state.split("_")]
            accumulated_degree = 0
            absolute_angle_of_arms = []
            for i in relative_angle_of_arms:
                accumulated_degree += i
                absolute_angle_of_arms.append(accumulated_degree)

            arm_with_degree = zip(base_end_points_of_arms, absolute_angle_of_arms)

            # remove the old arms
            for line in self.lines_to_delete_list:
                l=line.pop(0)
                l.remove()
            self.lines_to_delete_list = []

            translate = [0, 0]

            # go through the arm to draw them in the right position with the right angle
            for end_point, angle in tuple(arm_with_degree):
                rotated_end_point = self.rotate_vector(end_point, angle)

                start_point = translate
                end_point = translate + rotated_end_point

                numpy_row_stack = np.row_stack((start_point, end_point))

                self.lines_to_delete_list.append(self.ax.plot(numpy_row_stack[:, 0], numpy_row_stack[:, 1], marker='o', linewidth=5,  linestyle='-', color='b', markersize=10, markerfacecolor='r', markeredgecolor='k' ))
                translate = end_point

            if self.stop:
                break

            if self.quit:
                return

            plt.pause(self.pause_between_states)

        self.animate_in_progress = False

if __name__ == "__main__":

    goal_pos = [[94, 74], [96, 76]]
    blocker_position_list = [((-110,0), (-90,150)), ((75, 0), (96, 70)), ((75, 80), (96, 150)) ]
    state_list = ['0_0_0', '-20_120_-5', '-40_130_0']
    arms_list = [
            {"name": "arm0", "length": 100, "min_angle": 0, "max_angle": 90},
            {"name": "arm1", "length": 80, "min_angle": 0, "max_angle": 90},
            {"name": "arm2", "length": 50, "min_angle": 0, "max_angle": 90}
    ]

    my_robot_arm_animate = RobotArmAnimate()
    my_robot_arm_animate.animate(arms_list, state_list, goal_pos, blocker_position_list, pause_between_states=0.2)





