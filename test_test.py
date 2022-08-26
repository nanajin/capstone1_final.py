import numpy as np
import math
# def calculate_angle(point1, point2):
#     """ Calculate angle between two points """
#     ang_1 = np.arctan2(*point1[::-1])
#     ang_2 = np.arctan2(*point2[::-1])
#     return np.rad2deg((ang_1 - ang_2) % (2 * np.pi))
#
#
# a = np.array([14, 140])
# b = np.array([13, 120])
# c = np.array([12, 130])
# d = np.array([11, 110])
#
# # create vectors
# ba = a - b
# bc = c - b
# cd = d - c
#
# # calculate angle
# cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#
# angle = np.arccos(cosine_angle)
# inner_angle = np.degrees(angle)
#
# print
# inner_angle  # 8.57299836361
#
# # see how changing the direction changes the angle
# print
# calculate_angle(bc, cd)  # 188.572998364
# print
# calculate_angle(cd, bc)  # 171.427001636

# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

# a = Point(1,1)
# b = Point(3,5)
# c = Point(5,10)

def Angle(a1, b1, c1):
    a1[0] = a[0]-b[0]
    a1[1] = a[1]-b[1]
    c1[0] = c[0]-b[0]
    c1[1] = c[1]-b[1]
    b1[0] = 0
    b1[1] = 0

    theta = math.atan2(a1[1], a1[0]) - math.atan2(c1[1], c1[0])
    theta1 = theta * 180/np.pi
    print(theta1)
    # return a,b,c

Angle(a,b,c)
print(b)