import scipy.spatial
import numpy as np
import cv2
import scipy.misc
class addMatting():
    def __init__(self):
        # self.img_path = 'toy.jpg'
        # self.trimap_path = 'alpha4.png'
        return
        
    def addBoarder(self, img, inner = -2, outer = 6):
        '''
        it is used to generate Trimap image for the Mishima matting to work
        argvs: img_path: the path of image: eg. a.png
               inner: the width of adding stroke inside boarder
               outer: same, but outside
        '''
        # img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # draw counter, I need gray (127,127,127)
        cv2.drawContours(img,contours,inner,(127,127,127),outer)
        return img

    def mishima_matte(self, img, trimap):
        h,w,c = img.shape
        bg = trimap == 0
        fg = trimap == 255
        unknown = True ^ np.logical_or(fg,bg)
        fg_px = img[fg]
        bg_px = img[bg]
        unknown_px = img[unknown]
        print(fg_px)
        # Setup convex hulls for fg & bg
        fg_hull = scipy.spatial.ConvexHull(fg_px)
        fg_vertices_px = fg_px[fg_hull.vertices]
        bg_hull = scipy.spatial.ConvexHull(bg_px)
        bg_vertices_px = bg_px[bg_hull.vertices]

        # Compute shortest distance for each pixel to the fg&bg convex hulls
        d_fg = self.convex_hull_distance(fg_hull, unknown_px)
        d_bg = self.convex_hull_distance(bg_hull, unknown_px)


        # Compute uknown region alphas and add to known fg.
        alphaPartial = d_bg/(d_bg+d_fg)
        alpha = unknown.astype(float).copy()
        alpha[alpha !=0] = alphaPartial
        alpha = alpha + fg
        return alpha
    
    # Get fg/bg distances for each pixel from each surface on convex hull
    def convex_hull_distance(self, cvx_hull, pixels):
        d_hull = np.ones(pixels.shape[0]*cvx_hull.equations.shape[0]).reshape(pixels.shape[0],cvx_hull.equations.shape[0])*1000
        for j, surface_eq in enumerate(cvx_hull.equations):
            for i, px_val in enumerate(pixels):
                nhat= surface_eq[:3]
                d_hull[i,j] = np.dot(nhat, px_val) + surface_eq[3]
        return  np.maximum(np.amax(d_hull, axis=1),0)
'''
    # Load in image
    def main(self, img_path, trimap_path):    
        img  = scipy.misc.imread(img_path)
        trimap = scipy.misc.imread(trimap_path, flatten='True')

        alpha3 = mishima_matte(img, trimap)

        plt.imshow(alpha3, cmap='gray')
        plt.show()

    if __name__ == "__main__":
        import scipy.misc
        import matplotlib.pyplot as plt
        main(self.img_path, trimap_path)
'''