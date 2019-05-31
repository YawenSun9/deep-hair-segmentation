# h = [0,360], s = [0,1], v = [0,1]
import numpy as np
import cv2

class ChangeColor():
    def __init__(self):
        pass
    
    def RGBToHSV(self, R, G, B):
        h = s = v = 0.0
        r = R / 255.0
        g = G / 255.0
        b = B / 255.0
        mim_val = min(r,min(g,b))
        max_val = max(r,max(g,b))
        if max_val == mim_val:
            h = 0.0
        else:
            if max_val == r and g >= b:
                h = 60.0 * (g - b) / (max_val - mim_val)
            if max_val == r and g < b:
                h = 60.0 * (g - b) / (max_val - mim_val) + 360.0
            if max_val == g:
                h = 60.0 * (b - r) / (max_val - mim_val) + 120.0
            if max_val == b:
                h = 60.0 * (r - g) / (max_val - mim_val) + 240.0  
            if max_val == 0:
                s = 0.0
            else:
                s = (max_val - mim_val) / max_val
        v = float(max_val)
        return h, s, v

    def HSVToRGB(self, h, s, v):
        q = p = t = r = g = b = 0.0
        hN = 0
        if h < 0:
            h = 360 + h
        hN = h // 60
        p = v * (1.0 - s)
        q = v * (1.0 - (h / 60.0 - hN) * s)
        t = v * (1.0 - (1.0 - (h / 60.0 - hN)) * s)

        if hN == 0:
            r = v
            g = t
            b = p
        elif hN == 1:
            r = q
            g = v
            b = p
        elif hN == 2:
            r = p
            g = v
            b = t
        elif hN == 3:
            r = p
            g = q
            b = v
        elif hN == 4:
            r = t
            g = p
            b = v
        elif hN == 5:
            r = v
            g = p
            b = q
        else:
            pass
        
        R = int(self.clip((r * 255.0),0,255))
        G = int(self.clip((g * 255.0),0,255))
        B = int(self.clip((b * 255.0),0,255))
        
        return R, G, B
    
    def convert(self, h, s, v):
        '''
        most hair is black
        need high saturate and v
        '''
        h+=10
        
        s += 0.3
        s = self.clip(s,0,1)
        
        v += 0.3
        v = self.clip(v,0,1)
        return h, s, v
    
    # ----helper----
    def clip(self, n, lowerbound, upperbound):
        if n > upperbound:
            n = upperbound
        if n < lowerbound:
            n  = lowerbound
        return n
    
    def apply_mask(self, img, mask):
        '''
        input:
              mask: nparray, (H,W,C) 
        '''
        pass
        
'''
# demo
import matplotlib.pyplot as plt
%matplotlib inline

C = ChangeColor()
# h,s,v = C.RGBToHSV(1,1,1)
# print(h,s,v)
# h,s,v = C.convert(h,s,v)
# print(h,s,v)
# R, G, B = C.HSVToRGB(h,s,v)
# print(R,G,B)

img_path = 'toy.jpg'
mask_path = 'toymask_good.png'


img = cv2.imread(img_path)
mask = cv2.imread(mask_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.show()

# img = np.array([[[1,1,1],[23,45,145]]]) # shape (1,2,3)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (mask[i][j] == [255,255,255]).all():
            h,s,v = C.RGBToHSV(img[i][j][0],img[i][j][1],img[i][j][2])
            h,s,v = C.convert(h,s,v)
            img[i][j][0],img[i][j][1],img[i][j][2] = C.HSVToRGB(h,s,v)
# cv2.imwrite('img_change.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
# cv2.namedWindow('showimage')
# cv2.imshow("img", img)
# cv2.waitKey(0)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
'''
