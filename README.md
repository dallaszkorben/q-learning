# Q-learning
## Maze
## Robot arm

### Given elements
- **Arms**: 3 arms connected with joints. Their lengths and possible rotations are configured. The first arm is jointed to the ground.
- **Blockers**: The area which should not be crossed by any part of the arms
- **Goal position**: the dx,dy positions which should be reached by the end part of the last arm.
- **State**: The arm's relative rotations in order. For example "15_-123_32" referes to the arms where the 1st arm is rotated 15° the second is rotated -123° and the last arm is rotated 32° relative to the previous arm. 0° means up, -90° means left, 90° means right and 180° means down.
- **Actual position**: The end-point of the last arm. It can be calculated by the actual State and the length of the arms
- **Step Angle**: The precision of Angle in degree. 
- **Action**: In every State, 7 possible Actions are defined to bring it to the next State: 
  - 1st Arm +Angle°
  - 1st Arm -Angle°
  - 2nd Arm +Angle°
  - 2nd Arm -Angle°
  - 3rd Arm +Angle°
  - 3rd Arm -Angle°
  - Idle

### Used elements
- **Q table**: It is a Dict where the Key is the State, and the Value is a List, containing the belonging Action's calculated Q values. Not all combination of the Angles are in the key. The combinations which refere to an impossible or forbidden State are ignored.
- **Award table**: in this implementation there is no static Award table. It is supposed to be to tell which Action is allowed in a specific State and which is not having value 1 and which is the Goal State having value 999.

![Screenshot](wiki/robot_arm_animation.gif)

