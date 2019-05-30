'''
ref: https://blog.csdn.net/sunny2038/article/details/12889059
'''
import cv2
def addBoarder(img_path, inner=-5, outer=15):
    '''
    it is used to generate Trimap image for the Mishima matting to work
    argvs: img_path: the path of image: eg. a.png
           inner: the width of adding stroke inside boarder
           outer: same, but outside
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # draw counter, I need gray (127,127,127)
    cv2.drawContours(img,contours,inner,(127,127,127),outer)
    return img
'''
# demo
if __name__ == "__main__":
    img_path = 'alpha3.png'
    img = addBoarder(img_path, -5, 15)
    cv2.imshow("img", img)
    cv2.imwrite('alpha4.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
'''