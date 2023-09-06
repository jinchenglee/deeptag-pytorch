import numpy as np
import cv2

COLOR = (0, 255, 0)

img = np.load('./img.npy')
img.shape

# Load confidence prediction.
conf = np.load('grid_confidences_pred_np.npy')
cand_pos_list = np.where(conf[0,:,1]>0.8)[0]
# Load key points
grid_kpts_pred_np = np.load('grid_kpts_pred_np.npy')

# Draw points
for cand in cand_pos_list:
    (x, y) = grid_kpts_pred_np[0, cand, :]
    # Can only draw a integer position.
    img = cv2.circle(img, (int(x), int(y)), 1, COLOR, 2)

cv2.imshow('img', img)
cv2.waitKey(0)
#cv2.destroyAllWindows()

