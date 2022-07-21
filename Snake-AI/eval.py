import torch
import cv2
from model_tr import Net

PATH = "model/model.pth"
IMAGE_PATH = "cropped_images/croppedsavedImage_9.png"
img = cv2.imread(IMAGE_PATH)
print(img.shape)
img = cv2.resize(img, (200, 200))
filename = 'savedImage_test.png'
cv2.imwrite(filename, img)

# Model class must be defined somewhere
model = Net(input_size = (200,200,3), output_size = 26)
model.load_state_dict(torch.load(PATH))


shape_img = img.shape
img = img.reshape(shape_img[2], shape_img[0], shape_img[1])
img = torch.FloatTensor(img)
t = model(img)
print(img.shape)
print(t.shape)
move = torch.argmax(t).item()
print(move)