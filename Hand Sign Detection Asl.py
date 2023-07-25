import numpy as np  
import torch
import torch.nn
import torchvision 
from torch.autograd import Variable
from torchvision import transforms
import PIL 
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#This is the Label
Labels = { 0 : 'A',
           1 : 'B',
           2 : 'C',
           3 : 'D',
           4 : 'E',
           5 : 'F',
           6 : 'G',
           7 : 'H',
           8 : 'I',
           9 : 'K',
           10: 'L',
           11: 'M',
           12: 'N',
           13: 'O',
           14: 'P',
           15: 'Q',
           16: 'R',
           17: 'S',
           18: 'T',
           19: 'U',
           20: 'V',
           21: 'W',
           22: 'X',
           23: 'Y'
        }

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0,0.225])
    ]
) 


class Network(nn.Module): 
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(20, 30, 3) 
        self.dropout1 = nn.Dropout2d()
        
        self.fc3 = nn.Linear(30 * 3 * 3, 270) 
        self.fc4 = nn.Linear(270, 26) 
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
                
        x = x.view(-1, 30 * 3 * 3) 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return self.softmax(x)
    
    
    def test(self, predictions, labels):
        
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        print("Correct predictions: %5d / %5d (%5f)" % (correct, len(predictions), acc))
        
    
    def evaluate(self, predictions, labels):
                
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        
        acc = correct / len(predictions)
        return(acc)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation
model = Network() # initialize your model class
model.load_state_dict(torch.load('C:/Users/ishit/Downloads/sign_language_model.pth'))
print(model)
#model  = torch.load("C:/Users/ishit/Downloads/sign_language_model.pth") #Load model to CPU
#model  = model.to(device)   #set where to run the model and matrix calculation
#model.eval()                #set the device to eval() mode for testing




#Set the Webcam 
def Webcam_720p():
    cap.set(3,1280)
    cap.set(4,720)

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result,score





def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    image = image.convert("L")
    print(image)                             
    image = data_transforms(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    #image = image.cuda()
    #image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor
    
    
#Let's start the real-time classification process!
                                  
cap = cv2.VideoCapture(0) #Set the webcam
Webcam_720p()

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read() #Capture each frame
    
    
    if fps == 4:
        image        = frame[100:450,150:570]
        image_data   = preprocess(image)
        print(image_data)
        prediction   = model(image_data)
        result,score = argmax(prediction)
        fps = 0
        if result >= 0.5:
            show_res  = result
            show_score= score
        else:
            show_res   = "Nothing"
            show_score = score
        
    fps += 1
    cv2.putText(frame, '%s' %(show_res),(950,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.putText(frame, '(score = %.5f)' %(show_score), (950,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.rectangle(frame,(400,150),(900,550), (250,0,0), 2)
    cv2.imshow("ASL SIGN DETECTER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL SIGN DETECTER")