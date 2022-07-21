import cv2
image = cv2.imread("blank.png")
text = "AB"
coordinates = (70,260)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 5
color = (0,0,0)
thickness = 15
image = cv2.putText(image, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
filename = 'b.png'
cv2.imwrite(filename, image)