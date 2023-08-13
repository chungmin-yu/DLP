from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO

img = pd.read_csv('test_img.csv')
img_name = np.squeeze(img.values)
for index in range(len(img.values)):
    pimg = "data/"+ img_name[index]+ ".jpeg"
    with open(pimg, 'rb') as f:
        f = f.read()
        if f[-1] == 217 and f[-2] == 255:
            # EOI = \xff\xd9
            pass
        else:
            # borken image
            print(img_name[index])


# pimg = "data/"+ img_name[index]+ ".jpeg"
# with open(pimg, 'rb') as f:
#     f = f.read()
#     if f[-1] == 217 and f[-2] == 255:
#         # EOI = \xff\xd9
#         pass
#     else:
#         # borken image
#         print(img_name[index])
#         f = f+B'\xff'+B'\xd9'
# im = Image.open(BytesIO(f))
