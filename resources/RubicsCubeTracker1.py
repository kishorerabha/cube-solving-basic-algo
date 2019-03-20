import cv2
import numpy as np
from collections import OrderedDict

cap = cv2.VideoCapture(0)

while True:
    dictx = {} #store x coordinate of every cell with corresponding color as a dict
    dicty = {} #store y coordinate of every cell with correspinding color as a dict
    dict_xy = {} #store x,y coordinate of every cell, y refers x coordinate
    list3x3 = [] #
    list_coordinates_y_ordered = [] # store sorted y coordinates of every cell to find cell from up to bottom order
    y_coordinate_1st_row = []
    y_coordinate_2nd_row = []
    y_coordinate_3rd_row = []
    x_coordinate_1st_row = []
    x_coordinate_2nd_row = []
    x_coordinate_3rd_row = []
    kernel = np.ones((5,5),np.uint8)
    
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #masking--------------------------------
    #red color
    lower_red = np.array([150,50,60])
    upper_red = np.array([177,255,255])
    #blue color
    lower_blue = np.array([100,70,120])
    upper_blue = np.array([130,255,255])
    #yellow color
    lower_yel = np.array([30, 50, 100])
    upper_yel = np.array([50, 255, 255])
    #orange color
    lower_ora = np.array([177, 0, 150])
    upper_ora = np.array([180, 255, 255])
    lower_ora1 = np.array([0, 0, 150])
    upper_ora1 = np.array([30, 255, 255])
    
    lower_gre = np.array([50, 50, 110])
    upper_gre = np.array([85, 255, 255])
    #white color
    lower_whi = np.array([90, 60, 130])
    upper_whi = np.array([100,255 , 255])
    
    redmask = cv2.inRange(hsv, lower_red, upper_red)
    bluemask = cv2.inRange(hsv, lower_blue, upper_blue)
    yelmask = cv2.inRange(hsv, lower_yel, upper_yel)
    oramask1 = cv2.inRange(hsv, lower_ora, upper_ora)
    oramask2 = cv2.inRange(hsv, lower_ora1, upper_ora1)
    oramask = cv2.bitwise_or(oramask1,oramask2)
    
    
    gremask = cv2.inRange(hsv, lower_gre, upper_gre)
    whimask = cv2.inRange(hsv, lower_whi, upper_whi)
    
    #blurring
    medianblurred_red = cv2.medianBlur(redmask,15)
    medianblurred_blue = cv2.medianBlur(bluemask,15)
    medianblurred_yel = cv2.medianBlur(yelmask,15)
    medianblurred_ora = cv2.medianBlur(oramask,15)
    medianblurred_gre = cv2.medianBlur(gremask,15)
    medianblurred_whi = cv2.medianBlur(whimask,15)
    
    #erosion
    #medianblurred_red = cv2.erode(medianblurred_red,kernel,iterations = 2)
    #medianblurred_blue = cv2.erode(medianblurred_blue,kernel,iterations = 2)
    #medianblurred_yel = cv2.erode(medianblurred_yel,kernel,iterations = 2)
    #medianblurred_ora = cv2.erode(medianblurred_ora,kernel,iterations = 2)
    #medianblurred_gre = cv2.erode(medianblurred_gre,kernel,iterations = 2)
    #medianblurred_whi = cv2.erode(medianblurred_whi,kernel,iterations = 2)
    
    
    #contouring
    contours_red,_ = cv2.findContours(medianblurred_red,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_blue,_ = cv2.findContours(medianblurred_blue,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_yel,_ = cv2.findContours(medianblurred_yel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_ora,_ = cv2.findContours(medianblurred_ora,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_gre,_ = cv2.findContours(medianblurred_gre,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_whi,_ = cv2.findContours(medianblurred_whi,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    for contour_red in contours_red:
        area_red = cv2.contourArea(contour_red)
 
        if area_red > 6000 and area_red < 14000:
            rect_red = cv2.minAreaRect(contour_red)
            box_red = cv2.boxPoints(rect_red)
            box_red = np.int0(box_red)
            cv2.drawContours(frame, [box_red], -1, (0, 0, 255), 3)
            cv2.putText(frame,'RED',(box_red[0][0],box_red[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 1, cv2.LINE_8)
            dictx[box_red[0][0]] = 'red'
            dicty[box_red[0][1]] = 'red'
            dict_xy[box_red[0][1]] = box_red[0][0]
            
    for contour_blue in contours_blue:   
        area_blue = cv2.contourArea(contour_blue)
        
        if area_blue > 6000 and area_blue < 14000:
            rect_blue = cv2.minAreaRect(contour_blue)
            box_blue = cv2.boxPoints(rect_blue)
            box_blue = np.int0(box_blue)
            cv2.drawContours(frame, [box_blue], -1, (255, 0, 0), 3)
            cv2.putText(frame,'BLUE',(box_blue[0][0],box_blue[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 1, cv2.LINE_8)
            dictx[box_blue[0][0]] = 'blue'
            dicty[box_blue[0][1]] = 'blue'
            dict_xy[box_blue[0][1]] = box_blue[0][0]
            
    for contour_yel in contours_yel:   
        area_yel = cv2.contourArea(contour_yel)
        
        if area_yel > 6500 and area_yel < 14000:
            rect_yel = cv2.minAreaRect(contour_yel)
            box_yel = cv2.boxPoints(rect_yel)
            box_yel = np.int0(box_yel)
            cv2.drawContours(frame, [box_yel], -1, (0, 255, 255), 3)
            cv2.putText(frame,'YELLOW',(box_yel[0][0],box_yel[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 1, cv2.LINE_8)
            dictx[box_yel[0][0]] = 'yel'
            dicty[box_yel[0][1]] = 'yel'
            dict_xy[box_yel[0][1]] = box_yel[0][0]
            
    for contour_ora in contours_ora:   
        area_ora = cv2.contourArea(contour_ora)
        
        if area_ora > 6000 and area_ora < 14000:
            rect_ora = cv2.minAreaRect(contour_ora)
            box_ora = cv2.boxPoints(rect_ora)
            box_ora = np.int0(box_ora)
            cv2.drawContours(frame, [box_ora], -1, (0, 140, 255), 3)
            cv2.putText(frame,'ORANGE',(box_ora[0][0],box_ora[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,140,255), 1, cv2.LINE_8)
            dictx[box_ora[0][0]] = 'ora'
            dicty[box_ora[0][1]] = 'ora'
            dict_xy[box_ora[0][1]] = box_ora[0][0]
            
    for contour_gre in contours_gre:   
        area_gre = cv2.contourArea(contour_gre)
        
        if area_gre > 6000 and area_gre < 14000:
            rect_gre = cv2.minAreaRect(contour_gre)
            box_gre = cv2.boxPoints(rect_gre)
            box_gre = np.int0(box_gre)
            cv2.drawContours(frame, [box_gre], -1, (0, 255, 0), 3)
            cv2.putText(frame,'GREEN',(box_gre[0][0],box_gre[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 1, cv2.LINE_8)
            dictx[box_gre[0][0]] = 'gre'
            dicty[box_gre[0][1]] = 'gre'
            dict_xy[box_gre[0][1]] = box_gre[0][0]
            
    for contour_whi in contours_whi:   
        area_whi = cv2.contourArea(contour_whi)
        
        if area_whi > 6000 and area_whi < 14000:
            rect_whi = cv2.minAreaRect(contour_whi)
            box_whi = cv2.boxPoints(rect_whi)
            box_whi = np.int0(box_whi)
            cv2.drawContours(frame, [box_whi], -1, (255, 255, 255), 3)
            cv2.putText(frame,'LIGHT BLUE',(box_whi[0][0],box_whi[0][1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1, cv2.LINE_8)
            dictx[box_whi[0][0]] = 'whi'
            dicty[box_whi[0][1]] = 'whi'
            dict_xy[box_whi[0][1]] = box_whi[0][0]
            
    #displaying
    cv2.imshow('frame',frame)
    cv2.imshow('redmask',medianblurred_red)
    cv2.imshow('almostwhite',medianblurred_whi)
    #cv2.imshow('bluemaskblur',medianblurred_blue)
    #cv2.imshow('orangemask',medianblurred_ora)
    #cv2.imshow('greenmask',medianblurred_gre)
    
    #print(contours)
    
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k==32:
        if len(dictx) == 9 and len(dicty) == 9:
            #print(dictx)
            
            #How the code sorts the colours into a list from upper left
            #to lower right in the left to right manner for each row
            #--------------------------------------------------------
            #1) It first stores the x and y coordinates into seperate dicts 
            #   with the coordinates as index and the colours as value. The x and
            #   y coordinates are together stored in another dict with y coordinates
            #   as key and x coordinates as key.
            #2) It sorts the dict containing y coordinates so now we get a dict 
            #   where the 1st 3 colors belong to the 1st row, 2nd 3 colors
            #   belong to the 2nd row and last 3 colors to the 3rd row. So the
            #   dict is sorted row wise but not column wise yet.
            #3) We now store only the sorted y coordinates in a list. So now we 
            #   have a list where the 1st 3 elements/y coordinates belong to the 1st row, 2nd 
            #   3 elements belong to the 2nd row and last 3 elements to the 3rd 
            #   row. So the list is sorted row wise but not column wise yet.
            #4) We store the y coordinates of each row in 3 seperate lists for easy
            #   access.
            #5) We access the x coordinates from the dict containing both x and 
            #   y coordinates of each cell , using the y coordinates from the  
            #   above 3 lists and store them in 3 seperate lists. Now we have 3
            #   lists containing the x coordinates of every cell row wise
            #6) We now sort the 3 lists containing x coordinates row wise. Now
            #   we have 3 lists which are organised column wise. The 3 lists
            #   are already distinguished row wise.
            #7) Now we access the colors of the 1st row using the x coordinates
            #   which we sorted, from the dict containing the x coordinates of 
            #   each cell that we created in the beginning, and store them in a
            #   list.
            #8) Now we print the list.
            
            #--------------------------
            dicty_sorted = OrderedDict(sorted(dicty.items()))
            #print(dicty_sorted)
            
            #it iterates over the keys in dicty_sorted
            for y_coordinate in dicty_sorted:
                list_coordinates_y_ordered.append(y_coordinate)
            #print(list_1st_row_y)
            
            #---store y coordinates of 1st, 2nd and 3rd rows of cells
            for i in range(0,3):
                y_coordinate_1st_row.append(list_coordinates_y_ordered[i])
                
            for i in range(3,6):
                y_coordinate_2nd_row.append(list_coordinates_y_ordered[i])
                
            for i in range(6,9):
                y_coordinate_3rd_row.append(list_coordinates_y_ordered[i])
            #------------------------------------------------------------
            
            #---get x coordinates from y coordinates of each row
            for y_coordinate in y_coordinate_1st_row:
                x_coordinate_1st_row.append(dict_xy[y_coordinate])
                
            for y_coordinate in y_coordinate_2nd_row:
                x_coordinate_2nd_row.append(dict_xy[y_coordinate])
                
            for y_coordinate in y_coordinate_3rd_row:
                x_coordinate_3rd_row.append(dict_xy[y_coordinate])
            
            #-----sort x coordinates of each row
            x_coordinate_1st_row.sort()
            x_coordinate_2nd_row.sort()
            x_coordinate_3rd_row.sort()
            
            #_---------------------------
            
            #---store color from x coordinate
            for x_coordinate in x_coordinate_1st_row:
                list3x3.append(dictx[x_coordinate])
                
            for x_coordinate in x_coordinate_2nd_row:
                list3x3.append(dictx[x_coordinate])
                
            for x_coordinate in x_coordinate_3rd_row:
                list3x3.append(dictx[x_coordinate])
                
            print(list3x3)
            
            
            
        else:
            print('PRESS SPACE AGAIN')
        #dicty = {box_red[0][1]:'red',box_blue[0][1]:'blue',box_gre[0][1]:'green',box_yel[0][1]:'yellow',box_ora[0][1]:'orange',box_whi[0][1]:'white'}
        #dictx = {box_red[0][0]:'red',box_blue[0][0]:'blue',box_gre[0][0]:'green',box_yel[0][0]:'yellow',box_ora[0][0]:'orange',box_whi[0][0]:'white'}
        
        #ordered_dicty = OrderedDict(sorted(dicty.items()))
        #ordered_dictx = OrderedDict(sorted(dictx.items()))
        
        #list1strow = []
        

cv2.destroyAllWindows()
cap.release()