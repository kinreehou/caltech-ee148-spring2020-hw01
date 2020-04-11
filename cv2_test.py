import cv2

img = cv2.imread('../RedLights2011_Medium/RL-003.jpg')


#diff<0.01
#for y1,x1,y2,x2 in [[257, 400, 263, 404], [288, 138, 291, 140]]:
	#cv2.rectangle(img, (x1,y1-8), (x2+8,y2), (0,255,255),1)
	
#diff<0.02
for y1,x1,y2,x2 in [[203, 278, 208, 282], [216, 246, 219, 248], [221, 247, 230, 252], [237, 117, 241, 119], [253, 266, 256, 268], [261, 461, 274, 463], [266, 265, 269, 268]]:
	cv2.rectangle(img, (x1,y1-8), (x2+8,y2), (0,255,255),1)

	
cv2.imshow("detected",img)
cv2.waitKey(0)
cv2.destroyAllWindows()