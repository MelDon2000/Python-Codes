#!/usr/bin/env python
# coding: utf-8

# # **<center><font style="color:rgb(100,109,254)">Filipino Sign Language Interpreter Using Machine Learning</font> </center>**
# 
# <img src='https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/0fe6f706-efe9-4a81-b9b4-93079cce4da7/dczjk13-bbeb9ff1-e560-4b75-987c-10d0c74ca4d4.jpg/v1/fill/w_1024,h_1326,q_75,strp/filipino_sign_language_alphabet_by_jrdl30_dczjk13-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTMyNiIsInBhdGgiOiJcL2ZcLzBmZTZmNzA2LWVmZTktNGE4MS1iOWI0LTkzMDc5Y2NlNGRhN1wvZGN6amsxMy1iYmViOWZmMS1lNTYwLTRiNzUtOTg3Yy0xMGQwYzc0Y2E0ZDQuanBnIiwid2lkdGgiOiI8PTEwMjQifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.daSFktkN3JI_F1brN4g3FENqV0P3UugJyKHd7Ms-R3g'>
# 
# ## **<font style="color:rgb(134,19,348)"> Outline </font>**
# 
# - ***`Step 1:`*Perform Hands Landmarks Detection**
# 
# - ***`Step 2:`*Build the Fingers Counter**
# 
# - ***`Step 3:`*Visualize the Counted Fingers**
# 
# - ***`Step 4:`*Build the Hand Gesture Recognizer**
# 
# - ***`Step 5:`*Build a Selfie-Capturing System controlled by Hand Gestures**
# 

# In[ ]:





# In[ ]:





# ### **<font style="color:rgb(134,19,348)"> Import the Libraries</font>**
# 
# First, we will import the required libraries.
# 

# In[7]:


import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


# ### **<font style="color:rgb(134,19,348)">Initialize the Hands Landmarks Detection Model</font>**
# 
# After that, we will need to initialize the **`mp.solutions.hands`** class and then set up the **`mp.solutions.hands.Hands()`** function with appropriate arguments and also initialize **`mp.solutions.drawing_utils`** class that is required to visualize the detected landmarks. We will be working with images and videos as well, so we will have to set up the **`mp.solutions.hands.Hands()`** function two times. 
# 
# Once with the argument **`static_image_mode`** set to `True` to use with images and the second time **`static_image_mode`** set to `False` to use with videos. This speeds up the landmarks detection process, and the intuition behind this was explained in detail in the **[previous post](https://bleedai.com/real-time-3d-hands-landmarks-detection-hands-classification-with-mediapipe-and-python/)**.
# 

# In[8]:


# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


# ## **<font style="color:rgb(134,19,348)">Step 1: Perform Hands Landmarks Detection</font>**
# 
# 
# In the step, we will create a function **`detectHandsLandmarks()`** that will take an image/frame as input and will perform the landmarks detection on the hands in the image/frame using the solution provided by Mediapipe and will get **twenty-one 3D landmarks** for each hand in the image. The function will display or return the results depending upon the passed arguments.
# 
# <img src='https://drive.google.com/uc?export=download&id=1f2tcAy-efNBdN2FbaGTa0CdxiI4hIQ0O' width = 800>
# 
# The function is quite similar to the one in the **[previous post](https://bleedai.com/real-time-3d-hands-landmarks-detection-hands-classification-with-mediapipe-and-python/)**, so if you had read the post, you can skip this step. I could have imported it from a separate `.py` file, but I didn't, as I wanted to make this tutorial with the minimal number of prerequisites possible. 

# In[9]:


def detectHandsLandmarks(image, hands, draw=True, display = False):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''
    
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:
        
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(47,45,154), #red
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(217,218,220), #white
                                                                                     thickness=2, circle_radius=2))
            
        

       
    # Return the output image and results of hands landmarks detection.
    return output_image, results     


# In[10]:


def countFingers(image, results, draw=False):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''
    #the range value for error of margin that determines the upright and lateral position of fingers
    proposed_direction_margin = 0.08
    error_margin = 0.1
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB_LATERAL': False, 'RIGHT_INDEX_UPRIGHT': False, 'RIGHT_MIDDLE_UPRIGHT': False, 'RIGHT_RING_UPRIGHT': False,
                        'RIGHT_PINKY_UPRIGHT': False, 'LEFT_THUMB_LATERAL': False, 'LEFT_INDEX_UPRIGHT': False, 'LEFT_MIDDLE_UPRIGHT': False,
                        'LEFT_RING_UPRIGHT': False, 'LEFT_PINKY_UPRIGHT': False,
                        'RIGHT_THUMB_UPRIGHT': False, 'RIGHT_INDEX_LATERAL': False, 'RIGHT_MIDDLE_LATERAL': False, 'RIGHT_RING_LATERAL': False,
                        'RIGHT_PINKY_LATERAL': False, 'LEFT_THUMB_UPRIGHT': False, 'LEFT_INDEX_LATERAL': False, 'LEFT_MIDDLE_LATERAL': False,
                        'LEFT_RING_LATERAL': False, 'LEFT_PINKY_LATERAL': False
                        }
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y
                and hand_landmarks.landmark[tip_index - 2].y - hand_landmarks.landmark[tip_index].y > proposed_direction_margin
                and hand_landmarks.landmark[tip_index - 2].x - hand_landmarks.landmark[tip_index].x < error_margin
               ):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name+"_UPRIGHT"] = True
            
            # Check if the finger is lateral by comparing the x-coordinates of the tip and pip landmarks.
            if (hand_label=='Right' and hand_landmarks.landmark[tip_index].x < hand_landmarks.landmark[tip_index - 2].x
               and hand_landmarks.landmark[tip_index-2].x - hand_landmarks.landmark[tip_index].x > proposed_direction_margin
               and hand_landmarks.landmark[tip_index-2].y - hand_landmarks.landmark[tip_index].y < error_margin
               ):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name+"_LATERAL"] = True

            
            if (hand_label=='Left' 
               and hand_landmarks.landmark[tip_index].x > hand_landmarks.landmark[tip_index - 2].x      
               and hand_landmarks.landmark[tip_index].x - hand_landmarks.landmark[tip_index-2].x > proposed_direction_margin
               and hand_landmarks.landmark[tip_index-2].y - hand_landmarks.landmark[tip_index].y < error_margin
               ):
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name+"_LATERAL"] = True
                
        
        # Retrieve the x-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
        
        
        if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y):
            # Update the status of the finger in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB_UPRIGHT"] = True

            
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x) and (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x) and (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP - 2].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP - 2].x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB_LATERAL"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
            
        elif (hand_label=='Right' 
            and (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x) 
            and (thumb_tip_x > thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB_LATERAL"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
            
            
        elif (hand_label=='Left' 
            and (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP - 2].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP - 2].x) 
            and (thumb_tip_x < thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB_LATERAL"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
#    if draw:

        # Write the total count of the fingers of both hands on the output image.
        #cv2.putText(output_image, " Total Fingers: ", (50, 100),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        
#        font_size = 0.5
#        font_thickness = 1
#        i = 0
#        for line in fingers_statuses:
#            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
#            gap = 20

#            y = int((300 + textsize[1]) / 2) + i * gap
#            x = 10#for center alignment => int((img.shape[1] - textsize[0]) / 2)

#            cv2.putText(output_image, str(line) + "  " + str(fingers_statuses[line]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#                        font_size, 
#                        (255,255,255), 
#                        font_thickness, 
#                        lineType = cv2.LINE_AA)
#            i +=1
        #cv2.putText(output_image, str(fingers_statuses), (50,120), cv2.FONT_HERSHEY_SIMPLEX,1, (20,255,155), 10, 10)          
    # Return the output image, the status of each finger and the count of the fingers up of both hands.
    return output_image, fingers_statuses, count


# In[ ]:





# In[15]:


def recognizeGestures(image, fingers_statuses, count, results, still_J = False, draw=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was 
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''
    
    
    # Create a copy of the input image.
    output_image = image.copy()
    my_hand = results.multi_hand_landmarks[0]
    
    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT']
    
    # Initialize a dictionary to store the gestures of both hands in the image.
    if(still_J):
        hands_gestures = {'GESTURE': "LETTER J"}
    else:
        hands_gestures = {'GESTURE': "UNKNOWN"}
    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):
        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        

#fingers_statuses = {'RIGHT_THUMB_LATERAL': False, 'RIGHT_INDEX_UPRIGHT': False, 'RIGHT_MIDDLE_UPRIGHT': False, 'RIGHT_RING_UPRIGHT': False,
#                    'RIGHT_PINKY_UPRIGHT': False, 'LEFT_THUMB_LATERAL': False, 'LEFT_INDEX_UPRIGHT': False, 'LEFT_MIDDLE_UPRIGHT': False,
#                    'LEFT_RING_UPRIGHT': False, 'LEFT_PINKY_UPRIGHT': False}
        
    
  
 

        #[done] Check if the person is making the 'A' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_THUMB_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and fingers_statuses[hand_label+"_THUMB_LATERAL"] == False  
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x < my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_CMC].x < my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].x
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER A"
            
        #################################################################################################################### 
        
        
        
        # Check if the person is making the 'B' gesture with the hand.
        ####################################################################################################################
        
        # [done]Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == True            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False  
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER B"
            still_J=False
            
        ####################################################################################################################
        
        # Check if the person is making the 'C' gesture with the hand.
        ####################################################################################################################
        
        #[done] Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_THUMB_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == True
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x < my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y >0.2
             and (fingers_statuses[hand_label+"_RING_LATERAL"] == True 
                  or (my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y))
             and (fingers_statuses[hand_label+"_PINKY_LATERAL"] == True 
                  or (my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y > my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].y))
             and (fingers_statuses[hand_label+"_INDEX_LATERAL"] == True
                  or (my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y))
             
              
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER C"
            still_J=False
            
        ####################################################################################################################
        
        # Check if the person is making the 'D' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False  
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x - my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) < 0.03   
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)  
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER D"
            still_J=False
            
        ####################################################################################################################
        
        # Check if the person is making the 'E' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False  
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y < my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y > my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x < my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x  
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER E"
            
        ####################################################################################################################
        
                # Check if the person is making the 'F' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == True            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True     
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER F"
            still_J=False
            
        ####################################################################################################################
        
        # Check if the person is making the 'G' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == True
             and abs(my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y) < 0.08 
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER G"
            still_J=False
            
        ####################################################################################################################
        
        # Check if the person is making the 'H' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == True
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == True 
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < 0.1
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER H"
            still_J=False
            
        ####################################################################################################################
        
                # Check if the person is making the 'I' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and (fingers_statuses[hand_label+"_THUMB_LATERAL"] == True or fingers_statuses[hand_label+"_THUMB_UPRIGHT"] == True) 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x > results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            ):
            before_pinky_tip = float(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP].y) + 0.1
            hands_gestures['GESTURE'] = "LETTER I"
            placeholderframe, results = detectHandsLandmarks(frame, hands_videos, draw=True)
            if results.multi_hand_landmarks:
                placeholderframe, dynamicresults = detectHandsLandmarks(placeholderframe, hands_videos, draw=True)
                if(dynamicresults.multi_hand_landmarks == None):
                    after_pinky_tip = 0
                else:
                    after_pinky_tip = float(dynamicresults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP].y)
                # Update the gesture value of the hand that we are iterating upon to V SIGN.
                if( after_pinky_tip > before_pinky_tip ):
                    hands_gestures['GESTURE'] = "LETTER J"
                    still_J=True
            
        ####################################################################################################################
        
        
                        # Check if the person is making the 'J' gesture with the hand.
        ####################################################################################################################
        
                        ##OTHER ELSE
        ####################################################################################################################
        
                        # Check if the person is making the 'K' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == True
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False  
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == True     
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y < my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER K"
            still_J=False
            
        ####################################################################################################################
        
# Check if the person is making the 'L' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False  
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x - my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > 0.1
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER L"
            still_J=False
            
        ####################################################################################################################
        
# Check if the person is making the 'M' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger. 
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x < my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].x
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER M"
            still_J=False
        ####################################################################################################################       

        # Check if the person is making the 'N' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x < my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
             and (my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x - my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x) <0.03 
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER N"
            still_J=False
        ####################################################################################################################  
        
        # Check if the person is making the 'O' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False  
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) < 0.07 
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y) < 0.07
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y) < 0.07
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y - my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y) < 0.07
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x) < 0.1
             and abs(my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x - my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x) < 0.1
             
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER O"
            
        ####################################################################################################################
        
        
        
                                # Check if the person is making the 'P' gesture with the hand.
        ####################################################################################################################
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False  
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y 
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y  
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y < my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER P"
            still_J=False
        ####################################################################################################################
                                # Check if the person is making the 'Q' gesture with the hand.
        ####################################################################################################################
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False  
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y  
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y < my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y   
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER Q"
            still_J=False
        ####################################################################################################################
        
        # Check if the person is making the 'R' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False  
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False     
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER R"
            still_J=False
        ####################################################################################################################          
        
        # Check if the person is making the 'S' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_IP].y < results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
             and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP].y > results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP].y
             and results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x > results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x               
              ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER S"
            
        ####################################################################################################################        
                # Check if the person is making the 'T' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == True 
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y < my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
             and abs(my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y) < 0.15 
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER T"
            still_J=False
            
        #################################################################################################################### 
        
                # Check if the person is making the 'U' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < 0.05
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER U"
            still_J=False
            
        #################################################################################################################### 
        
                        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > 0.1
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER V"
            still_J=False
            
        #################################################################################################################### 
        
                                # Check if the person is making the 'W' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == True            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > 0.1
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x - my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > 0.1
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER W"
            still_J=False
            # Update the color value to green.
            color=(0,255,0)
            
        ####################################################################################################################
        
                                        # Check if the person is making the 'X' gesture with the hand.
        ####################################################################################################################
        elif  (fingers_statuses[hand_label+"_INDEX_LATERAL"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].y
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
             and abs(my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y) < 0.15 
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER X"
            still_J=False
            
        ####################################################################################################################
        
                                        # Check if the person is making the 'Y' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
             and fingers_statuses[hand_label+"_THUMB_UPRIGHT"] == True
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x - my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > 0.1  
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "LETTER Y"
            still_J=False
            # Update the color value to green.
            color=(0,255,0)
            
        ####################################################################################################################
        # Check if the person is making the 'Z' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False  
             and my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x   
             and hands_gestures['GESTURE'] != "LETTER Z"
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "Z ATTEMPT"
            still_J=False    
            before_index_tip = float(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x) + 0.008
            placeholderframe, results = detectHandsLandmarks(frame, hands_videos, draw=True)
            if results.multi_hand_landmarks:
                time.sleep(0.02)
                placeholderframe, dynamicresults = detectHandsLandmarks(placeholderframe, hands_videos, draw=True)
                if(dynamicresults.multi_hand_landmarks == None):
                    after_index_tip = 0
                else:
                    after_index_tip = float(dynamicresults.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
                if( after_index_tip > before_index_tip):
                    hands_gestures['GESTURE'] = "LETTER Z"
            
        ####################################################################################################################      
    
    
    
    
    
    
    
    
    
    
                                        # Check if the person is making the 'Y' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_THUMB_UPRIGHT"] == True
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x - my_hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x > 0.1  
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "I LOVE YOU"
            still_J=False
            # Update the color value to green.
            color=(0,255,0)
            
        ####################################################################################################################




                                         # Check if the person is making the 'FU' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == True
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "VERY BAD!!"
        ####################################################################################################################       
        
                                # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > my_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x             
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x > my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x > my_hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x             
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_MCP].x > my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].x
             and my_hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x > my_hand.landmark[mp_hands.HandLandmark.PINKY_PIP].x
               and my_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y > my_hand.landmark[mp_hands.HandLandmark.THUMB_IP].y
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "THUMBS UP"
            
        
            
        #################################################################################################################### 
        
                                        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == True
             and fingers_statuses[hand_label+"_RING_LATERAL"] == True            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == True 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == True
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "WORD DELETE"
            
        
            
        #################################################################################################################### 
        
                
                                        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == True
             and fingers_statuses[hand_label+"_RING_LATERAL"] == True            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == True 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "WORD RESET"
            
        
            
        #################################################################################################################### 
                                                # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == True            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == True 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "SENTENCE DELETE"
            
        
            
        #################################################################################################################### 
        
                                                # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == False            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == True 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "SENTENCE RESET"
        #################################################################################################################### 
                
                                                # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif  (fingers_statuses[hand_label+"_INDEX_UPRIGHT"] == False 
             and fingers_statuses[hand_label+"_MIDDLE_UPRIGHT"] == False
             and fingers_statuses[hand_label+"_RING_UPRIGHT"] == True            
             and fingers_statuses[hand_label+"_PINKY_UPRIGHT"] == True 
             and fingers_statuses[hand_label+"_MIDDLE_LATERAL"] == False
             and fingers_statuses[hand_label+"_RING_LATERAL"] == False            
             and fingers_statuses[hand_label+"_PINKY_LATERAL"] == False 
             and fingers_statuses[hand_label+"_INDEX_LATERAL"] == False
            ):
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures['GESTURE'] = "2 JHONGS"
        #################################################################################################################### 
 
        
    # Check if the hands gestures are specified to be written.
    if(hands_gestures['GESTURE'] == "LETTER I" and still_J):
        hands_gestures['GESTURE'] = "LETTER J"
    if(hands_gestures['GESTURE'] == "UNKNOWN" and still_J):
        hands_gestures['GESTURE'] = "LETTER J"
    if(hands_gestures['GESTURE'] == "LETTER A" and still_J):
        hands_gestures['GESTURE'] = "LETTER J"
    

        
    
    
    # Return the output image and the gestures of the both hands.
    return output_image, hands_gestures, still_J


# In[ ]:





# In[34]:


# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

# Create named window for resizing purposes.
cv2.namedWindow('FILIPINO SIGN LANGUAGE ALPHABET INTERPRETER', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('FILIPINO SIGN LANGUAGE ALPHABET INTERPRETER',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

color = (255,255,255)
still_J = False
gesture_before = "UNKNOWN"
gesture_locked_in_counter = 0
gesture_word = []
word_copy = []
gesture_sentence = []
hands_gestures = {'GESTURE': "UNKNOWN"}
ok_counter = 0
# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    frame, results = detectHandsLandmarks(frame, hands_videos, draw=True)
    ksize = (20, 20)
    frame = cv2.blur(frame, ksize) 
    if results.multi_hand_landmarks:
        if(ok_counter>0):
            y = int(frame_height)  #int((300 + textsize[1]) / 2) + i * gap
            x = (30*ok_counter) 
                    #for center alignment => int((img.shape[1] - textsize[0]) / 2) 
            cv2.rectangle(frame,(0,int(int(frame_height)*.9)),(x,y),(57,45,34),-1) 
        ok_counter+=1
        ksize = (10, 10)
        frame = cv2.blur(frame, ksize) 
        if ok_counter >= 50:
            break  
    else:
        ok_counter = 0
    welcome_x = int(int(frame_width)*.08)
    welcome_y = int(int(frame_height) / 2)
    # Perform Hands landmarks detection on the frame.
    cv2.putText(frame, "WELCOME!", (welcome_x, welcome_y), cv2.FONT_HERSHEY_SIMPLEX, 7 , (255,255,255), 3, lineType = cv2.LINE_AA)
    cv2.imshow('FILIPINO SIGN LANGUAGE ALPHABET INTERPRETER', frame)
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    cv2.waitKey(1) & 0xFF




while camera_video.isOpened():
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos, draw=True)
    
    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
            
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, draw=True)
        
        # Perform the hand gesture recognition on the hands in the frame.
        notframe, hands_gestures, still_J = recognizeGestures(frame, fingers_statuses, count, results, still_J, draw=True)    
         
        if gesture_before == "LETTER Z" and (hands_gestures['GESTURE'] == "Z ATTEMPT" or hands_gestures['GESTURE'] == "UNKNOWN"):
            hands_gestures['GESTURE'] = "LETTER Z"       
        gesture_before = hands_gestures['GESTURE']    
        # Display the frame.
        if(hands_gestures['GESTURE']=="VERY BAD!!"):
            coord1=(int(frame_width * results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP].x), int(frame_height * results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y))
            coord2=(int(frame_width * results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_CMC].x), int(frame_height * results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y))
            thickness=10
            bad = cv2.rectangle(frame,coord1,coord2,(0,0,255),thickness)
            ksize = (50, 50)
            frame = cv2.blur(bad, ksize) 

        # Write the hand gesture on the output image. 
        cv2.rectangle(frame,(0 ,frame_height),(frame_width, int(int(frame_height)*.82)),(57,45,34),-1) 
        font_size = 3
        font_thickness = 2
        i = 0   
        if hands_gestures['GESTURE'] == "UNKNOWN":
            color = (0,0,255)
        else:
            color = (0,255,0)
            
        cv2.putText(frame,"GESTURE:", (int(int(frame_width)*.03) , int(int(frame_height)*.95)), cv2.FONT_HERSHEY_SIMPLEX,font_size, (255,255,255),font_thickness, lineType = cv2.LINE_AA)
        cv2.putText(frame,str(hands_gestures['GESTURE']), (int(int(frame_width)*.38), int(int(frame_height)*.95)), cv2.FONT_HERSHEY_SIMPLEX,font_size, color,font_thickness, lineType = cv2.LINE_AA)           
        cv2.putText(frame, "HAND DETECTED" , (int(int(frame_width)*.05), int(int(frame_height)*.87)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        #cv2.putText(frame,str(float(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_IP].visibility)), (10, 940), cv2.FONT_HERSHEY_SIMPLEX,3, (255,255,255),3, lineType = cv2.LINE_AA)           

    else:
        hands_gestures = {'GESTURE': "UNKNOWN"}
    
    
    if gesture_before == hands_gestures['GESTURE'] and hands_gestures['GESTURE'] != "UNKNOWN":
        gesture_locked_in_counter+=1
        if gesture_locked_in_counter > 30:
            if (hands_gestures['GESTURE'] == "THUMBS UP" 
                and hands_gestures['GESTURE'] != "WORD DELETE"
                and hands_gestures['GESTURE'] != "WORD RESET"
                and hands_gestures['GESTURE'] != "SENTENCE DELETE"
                and hands_gestures['GESTURE'] != "SENTENCE RESET"
                and hands_gestures['GESTURE'] != "VERY BAD!!"
                and hands_gestures['GESTURE'] != "2 JHONGS"
                and hands_gestures['GESTURE'] != "Z ATTEMPT"):
                gesture_word.append(" ")
                word_copy = gesture_word[:]
                gesture_sentence.append(word_copy)
                gesture_word.clear()
                gesture_locked_in_counter=0
            elif hands_gestures['GESTURE'] == "WORD DELETE" and len(gesture_word)> 0 :
                toDelete = gesture_word.pop(-1)   
                gesture_locked_in_counter=0
            elif hands_gestures['GESTURE'] == "WORD RESET" and len(gesture_word)> 0 :
                toDelete = gesture_word.clear()  
                gesture_locked_in_counter=0
            elif hands_gestures['GESTURE'] == "SENTENCE DELETE" and len(gesture_sentence)> 0 :
                toDelete = gesture_sentence.pop(-1)   
                gesture_locked_in_counter=0
            elif hands_gestures['GESTURE'] == "SENTENCE RESET" and len(gesture_sentence)> 0 :
                toDelete = gesture_sentence.clear()  
                gesture_locked_in_counter=0 
            elif (hands_gestures['GESTURE'] != "WORD DELETE"
                and hands_gestures['GESTURE'] != "WORD RESET"
                and hands_gestures['GESTURE'] != "SENTENCE DELETE"
                and hands_gestures['GESTURE'] != "SENTENCE RESET"
                and hands_gestures['GESTURE'] != "VERY BAD!!"
                and hands_gestures['GESTURE'] != "2 JHONGS"
                and hands_gestures['GESTURE'] != "Z ATTEMPT"):
                letter = hands_gestures['GESTURE'].split()
                gesture_word.append(letter[1]) 
                gesture_locked_in_counter = 0
    else:
        gesture_locked_in_counter=0
        
    
    font_size = 3
    font_thickness = 2
    wordspace = 0
    if len(gesture_word) > 0 :
        for letter in gesture_word:
            textsize = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
            gap = 10
            y = int(frame_height * .1)  #int((300 + textsize[1]) / 2) + i * gap
            rect_coord_1 = wordspace , 0
            x = (int(frame_width * .01) + wordspace) 
                    #for center alignment => int((img.shape[1] - textsize[0]) / 2)
            wordspace += textsize[0] + gap    
  
            rect_coord_2 = int(wordspace+gap), int(frame_height * .12) 
            cv2.rectangle(frame,rect_coord_1,rect_coord_2,(57,45,34),-1) 
            cv2.putText(frame, str(letter) , (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, 
                        (255,255,255), 
                        font_thickness, 
                        lineType = cv2.LINE_AA)

          
    sentence_wordspace = 0
    sentence_wordspace_check = 0
    multiliner = 0
    gap = 10
    textsize = (0,0)
    if len(gesture_sentence) > 0 :
        sentence_wordspace = sentence_wordspace_check 
        for words in gesture_sentence: 
            for letters in words:
                wordtextsize = cv2.getTextSize(letters, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
                sentence_wordspace_check += wordtextsize[0]    
                
                if sentence_wordspace_check > int(int(frame_width)-200):
                    multiliner += 1
                    sentence_wordspace = 0
                    sentence_wordspace_check = 0
                    break
                    
            for letters in words:
                sentencetextsize = cv2.getTextSize(letters, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
                y = int(frame_height * .6) + (multiliner * 85)  #int((300 + textsize[1]) / 2) + i * gap
                rect_coord_1 = sentence_wordspace , int(y-(sentencetextsize[1] + 10))
                x = (int(frame_width * .01) + sentence_wordspace) 
                        #for center alignment => int((img.shape[1] - textsize[0]) / 2)
                sentence_wordspace += sentencetextsize[0] + gap    
                
                rect_coord_2 = int(sentence_wordspace+gap), int(y+10) 
                cv2.rectangle(frame,rect_coord_1,rect_coord_2,(57,45,34),-1) 
                cv2.putText(frame, str(letters) , (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, 
                            (255,255,255), 
                            font_thickness, 
                            lineType = cv2.LINE_AA)
        
        
        
        
        
        
        
    #cv2.putText(frame,str(gesture_word), (100, 340), cv2.FONT_HERSHEY_SIMPLEX,2, (255,255,255),3, lineType = cv2.LINE_AA)        
    #cv2.putText(frame,str(gesture_word), (100, 540), cv2.FONT_HERSHEY_SIMPLEX,8, (255,255,255),3, lineType = cv2.LINE_AA)
    cv2.imshow('FILIPINO SIGN LANGUAGE ALPHABET INTERPRETER', frame)
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()


# ## 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




