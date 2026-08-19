import requests 
import cv2 
import numpy as np 


url = "http://localhost:8080/api/v1/edit" 

with open("test_image/sample.jpg" , "rb") as f:
    file = {"file" : ("sample.jpg" , f , "image/jpeg")}
    data = {"feature" : "nose" , "intensity": "0.5"} 
    
    respose = requests.post(url , file=file , data=data) 
    
if respose.staus_code == 200 :
    with open("result.jpg" , "wb") as f :
        f.write(respose.content)
    print("✅ تصویر تغییر یافته ذخیره شد: result.jpg")
else:
    print("❌ خطا:" , respose.json())
    