def predict(boxes, img_path)

	img = cv2.imread(img_path)

	for y1,x1,y2,x2 in boxes_from_red:
		cv2.rectangle(img, (x1,y1-8), (x2+8,y2), (0,255,255),2)

	cv2.imshow("detected",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
