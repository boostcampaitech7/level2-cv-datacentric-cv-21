import cv2


img = cv2.imread('data/chinese_receipt/img/train/extractor.zh.in_house.appen_000326_page0001.jpg')

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path = "LapSRN_x8.pb"
 
sr.readModel(path)
 
sr.setModel("lapsrn",8)
 
result = sr.upsample(img)
 
# Resized image
resized = cv2.resize(img,dsize=None,fx=8,fy=8)


cv2.imwrite('/data/ephemeral/home/github/level2-cv-datacentric-cv-21/result/LapSRN_x8_zh00326.jpg', img)
cv2.imwrite('/data/ephemeral/home/github/level2-cv-datacentric-cv-21/result/resized_zh00326.jpg', resized)