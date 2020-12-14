import torch
import numpy as np
import math
import random
from PIL import Image

import os
from datetime import datetime
# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    try:
        f.write(f'{str(datetime.now())}: {s}\n')
    except Exception:
        f.write('Print error\n')
    f.close()

def format_losses(loss, split='train'):
    log_string = ' '
    log_string += f'{split} loss ['
    log_string += f'depth: {loss:.5f}'
    log_string += ']'
    return log_string

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

# From https://www.programmersought.com/article/16104942640/
class AddPepperNoise(object):
    """Increase salt and pepper noise
    Args:
        snr （float）: Signal Noise Rate
                 p (float): probability value, perform the operation according to probability
    """
    # The default signal-to-noise ratio is 90%, and 90% of the pixels are the original image
    def __init__(self, snr, p=0.9, seed=1):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p
        random.seed(seed)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # Set the percentage of the signal SNR
            signal_pct = self.snr
            # Percentage of noise
            noise_pct = (1 - self.snr)
            # Select the mask mask value 0, 1, 2 0 represents the original image 1 represents salt noise 2 represents pepper noise
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # Salt noise white
            img_[mask == 2] = 0     # Pepper noise black
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

if __name__ == '__main__':
    print(cal_relative_pose(np.array([-5.902101516723632812e+01, 1.927141571044921875e+02, 5.996415138244628906e+00, -2.683681845664978027e-01, 2.737782299518585205e-01, 6.468879580497741699e-01, 6.592115163803100586e-01]), np.array([-5.506656265258789062e+01, 1.961531982421875000e+02, 5.998308658599853516e+00,-4.198011383414268494e-02,3.792373836040496826e-01, 1.026619002223014832e-01,9.186278581619262695e-01])))