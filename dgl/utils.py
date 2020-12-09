import torch
import numpy as np
import math


def quat_to_so3(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    # a_row_column
    a00 = 1 - 2 * y ** 2 - 2 * z ** 2
    a10 = 2 * x * y + 2 * z * w
    a20 = 2 * x * z - 2 * y * w
    a01 = 2 * x * y - 2 * z * w
    a11 = 1 - 2 * x ** 2 - 2 * z ** 2
    a21 = 2 * y * z + 2 * x * w
    a02 = 2 * x * z + 2 * y * w
    a12 = 2 * y * z - 2 * x * w
    a22 = 1 - 2 * x ** 2 - 2 * y ** 2
    return np.array([a00, a10, a20, a01, a11, a21, a02, a12, a22])

def cal_relative_pose(p1, p2):
    #print('p1:' , p1, '   p2: ' ,p2)
    p1t,p2t,p1r,p2r = p1[:3], p2[:3],p1[3:], p2[3:]
    x1, y1, z1, w1 = -p1r[0], -p1r[1], -p1r[2], p1r[3] # vector conjugate of q1
    x2, y2, z2, w2 = p2r[0], p2r[1], p2r[2], p2r[3]
    # q2 * q1.inverse
    rp = quat_to_so3([(w1*x2 + x1*w2 + y1*z2 - z1*y2), (w1*y2 - x1*z2 + y1*w2 + z1*x2), (w1*z2 + x1*y2 - y1*x2 + z1*w2) , (w1*w2 - x1*x2 - y1*y2 - z1*z2)])[:6]
    return np.concatenate((p2t-p1t, rp))
if __name__ == '__main__':
    print(cal_relative_pose(np.array([-5.902101516723632812e+01, 1.927141571044921875e+02, 5.996415138244628906e+00, -2.683681845664978027e-01, 2.737782299518585205e-01, 6.468879580497741699e-01, 6.592115163803100586e-01]), np.array([-5.506656265258789062e+01, 1.961531982421875000e+02, 5.998308658599853516e+00,-4.198011383414268494e-02,3.792373836040496826e-01, 1.026619002223014832e-01,9.186278581619262695e-01])))