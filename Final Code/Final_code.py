# -----------------------Imports-----------------------
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
#import imutils
import matplotlib.image as mpimg
import random
import math
from keras.applications.resnet50 import ResNet50
# importing only those functions
# which are needed
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import PIL.Image, PIL.ImageTk


resnet50 = ResNet50(weights='imagenet', include_top=False)


def call_model():  # call the saved model
    pipeline = pickle.load(open("pipeline.pickle", 'rb'))
    return pipeline


def sortGateX1(elem):
    return elem[2]


def sortLineX1(elem):
    return elem[3]


def segment(path):
    pipeline = call_model()
    img = cv2.imread(path)
    white = (np.copy(img)) * 0 + 255
    original = (np.copy(img))
    lines_img = (np.copy(img))
    final_image = (np.copy(img))

    gates = []
    array_gates = []
    id_g = 0
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarization (thresholding)
    ret, thresh_1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # Denoizing (Removing noise)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh_1, kernel, iterations=1)
    # cv.imshow("", erosion)

    # ------SEGMENTATION------

    # 1. canny edge detection
    edged = cv2.Canny(erosion, 30, 200)
    # 2. Finding contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    num_rect = 1

    # print the area of each contour
    i = 1
    total_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        total_area = total_area + area
        print("Area " + str(i) + "-->" + str(area))
        i = i + 1

    mean_area = total_area / len(contours)

    for c in contours:
        # for all the contours do the following
        # if area of the contour is smaller than mean area don't do anything
        if cv2.contourArea(c) <= mean_area:
            continue

        '''if area of the contour is larger than mean area find the best rectangle bounding this contour,
         and find it's coordinates'''

        x, y, w, h = cv2.boundingRect(c)
        # rect=cv2.rectangle(img, (x - 12, y - 12), (x + w + 12, y + h + 12), (0, 255,0), 4) # 4 is the line's thickness
        new_img_2 = img.copy()  # make a copy of the original image to work on it

        lines_img[y - 20: y + h + 20, x - 20: x + w + 20] = white[y - 20: y + h + 20, x - 20: x + w + 20]

        new_img_2 = new_img_2[y - 20: y + h + 20,
                    x - 20: x + w + 20]  # crop the image in the place of the rectabgle contour

        # --------------------------------
        # prediction part
        new_img_2 = cv2.resize(new_img_2, (224, 224), interpolation=cv2.INTER_AREA)
        img_data = image.img_to_array(new_img_2)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        resnet_features = resnet50.predict(img_data)
        go = []
        go.append(resnet_features.flatten())
        preds = pipeline.predict(go)

        x1_g = x - 20
        y1_g = y - 20
        x2_g = x - 20
        y2_g = y + h + 20
        x3_g = x + w - 20
        y3_g = y - 20
        x4_g = x + w + 20
        y4_g = y + h + 20
        id_g = id_g + 1

        if preds[0] == 'And':
            array_gates.append(['A', id_g, x1_g, y1_g, x2_g, y2_g, x3_g, y3_g, x4_g, y4_g])
            gates.append([id_g, 'A'])
        else:
            array_gates.append(['O', id_g, x1_g, y1_g, x2_g, y2_g, x3_g, y3_g, x4_g, y4_g])
            gates.append([id_g, 'O'])

        # ---------------Get End Points--------------

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        cv2.drawContours(original, [c], -1, (0, 255, 255), 2)
        cv2.circle(original, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(original, extRight, 8, (0, 255, 0), -1)
        cv2.circle(original, extTop, 8, (255, 0, 0), -1)
        cv2.circle(original, extBot, 8, (255, 255, 0), -1)

        # -------------END of End Points----------------
        '''
        plt.imshow(new_img_2)
        plt.text(10, 180, 'Predicted: %s' % preds[0], color='k', backgroundcolor='red', alpha=0.8)
        plt.show()
        '''
        #
        # -------------------------------
        center = (x, y)
        print("center -- >", center)
        num_rect = num_rect + 1

    print("Number of rectangles = " + str(num_rect))
    print("Mean area = ", mean_area)
    print('Numbers of contours found=' + str(len(contours)))
    '''
    plt.imshow(original)
    plt.show()
    plt.imshow(lines_img)
    plt.show()
    '''
    # Preprocessing part for lines
    ret, thresh_1 = cv2.threshold(lines_img, 70, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)  # note this is a horizontal kernel
    e_im = cv2.erode(thresh_1, kernel, iterations=3)
    d_im = cv2.dilate(e_im, kernel, iterations=2)
    #plt.imshow(d_im)
    #plt.show()

    # -------------------------------END OF COMPONENTS SEGMENTATION-------------------
    # --------------------------------------------------------------------------------
    # ---------------------------LINES COORDINATES------------------------------------

    original = (np.copy(d_im))
    array_line = []
    id_l = 'a'
    # 1. canny edge detection
    edged = cv2.Canny(original, 30, 200)
    # 2. Finding contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_area = 0
    mean_area = 0
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        total_area = total_area + area
        print("Area " + str(i) + "-->" + str(area))
        i = i + 1
    mean_area = total_area / i
    for c in contours:
        if cv2.contourArea(c) < 0.9:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.rectangle(img, (x - 12, y - 12), (x + w + 12, y + h + 12), (0, 255, 0),
                             4)  # 4 is the line's thickness
        # ---------------Get End Points--------------

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        cv2.drawContours(original, [c], -1, (0, 255, 255), 2)
        cv2.circle(original, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(original, extRight, 8, (0, 255, 0), -1)

        x1_l = extRight[0]
        x2_l = extLeft[0]
        y1_l = extRight[1]
        y2_l = extLeft[1]
        array_line.append([id_l, x1_l, y1_l, x2_l, y2_l])
        id_l = chr(ord(id_l) + 1)

    #plt.imshow(original)
    #plt.show()
    # -------------END of End Points----------------
    # ---------------------------------------------------------------
    # show the image

    new_img_for_gui = np.copy(final_image)
    for i in array_line:
        #plt.imshow(final_image)
        #plt.text(i[3], i[4], '%s' % i[0], color='k', backgroundcolor='red', alpha=0.8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_img_for_gui, str(i[0]), (i[3], i[4]), font, 2, (0, 0, 255), 5, cv2.LINE_AA)
    '''
    plt.show()
    plt.imshow(new_img_for_gui)
    plt.show()
    '''

    # ----------------------------------------------------------------
    # TESTING PART

    array_gates.sort(key=sortGateX1)
    array_line.sort(key=sortLineX1)

    print(" ----------- GATES COORDINATES -----------")
    for coor_gate in array_gates:
        print(coor_gate)

    print(" ----------- LINES COORDINATES -----------")

    new_arr = array_line
    array_line = []
    for coor_line in new_arr:
        if abs(coor_line[1] - coor_line[3]) > 50:
            array_line.append(coor_line)

    for coor_line in array_line:
        print(coor_line)
    # ---------------------------------------------------------------

    gates, output = check_line_connect_gate(array_gates, array_line, gates)

    equ = ""
    equ = get_equ(gates, output)
    print(output, "=", equ)
    equ = output, "=", equ

    return equ, array_line, final_image, output


# ----------------------------------------------------------------
def check_line_connect_gate(array_gates, array_line, final_gates):
    # final_gates [gate Id , gate type , in 1 ,in 2 , out , Rule]

    for line in array_line:
        for gate in array_gates:
            distance_In_1 = math.sqrt(((gate[2] - line[1]) ** 2) + ((gate[3] - line[2]) ** 2))
            distance_In_2 = math.sqrt(((gate[4] - line[1]) ** 2) + ((gate[5] - line[2]) ** 2))
            '''
            distance_out_1 = math.sqrt(((gate[6] - line[3]) ** 2) + ((gate[7] - line[4]) ** 2))
            distance_out_2 = math.sqrt(((gate[8] - line[3]) ** 2) + ((gate[9] - line[4]) ** 2))
            '''
            middle_y = (gate[7] + gate[9]) / 2
            middle_x = (gate[6] + gate[8]) / 2
            distance_out = math.sqrt(((middle_x - line[3]) ** 2) + ((middle_y - line[4]) ** 2))
            # print (line[0] , distance_In_1 , distance_In_2 ,distance_out_1 ,distance_out_2)

            index = gate[1] - 1
            # right part start point
            if distance_In_1 < 120 or distance_In_2 < 120:
                final_gates[index].append(line[0])

            # left part end point
            if distance_out < 150:
                final_gates[index].append(line[0])

    print("-------------------------------------------------------")
    print("ARRAY OF CONNECTIONS --NEW PART--")
    for gat in final_gates:
        print(gat)

    return final_gates, array_line[len(array_line) - 1][0]


def get_equ(gates, output):
    equ = ""
    index = len(gates) + 1

    for i in gates:
        if i[4] == output:
            index = gates.index(i)

    if index == len(gates) + 1:
        equ = equ + str(output)

    else:
        equ = equ + " ( " + get_equ(gates, gates[index][2])

        if gates[index][1] == 'A':
            equ = equ + " & "
        else:
            equ = equ + " | "

        equ = equ + get_equ(gates, gates[index][3]) + " ) "

    return equ


# --------------------------------------------------------
# --------------------VERILOG-----------------------------
def verilog(equation,output):
    input = ''
    for element in equation:
        input = input + element
    equ = input
    input = input.replace(' ( ', '')
    input = input.replace(' ) ', '')
    input = input.replace(' = ', '')
    input = input.replace(' & ', '')
    input = input.replace(' | ', '')
    input = input[2:]
    inpu = ''
    for element in input:
        inpu = inpu + element + ', '
    code = "module verilog_gates( input " + inpu + "; output " + output + ");"
    code2 = "\nassign " + equ
    code3 = "\nendmodule"

    f = open("verilog.v", "w")
    f.write(code + code2 + code3)
    f.close()

# ----------------------------------------------------------------


'''
def main():
    exit = ""
    while exit != "0":
        path = input("Enter Images path: ")
        print("Output equation", segment(path))
        exit = input("do you want to exit: ")


main()
'''
# ------------------------------------------------------------------------------------
# ----------------------------------GUI PART------------------------------------------
# ------------------------------------------------------------------------------------


# creating tkinter window
root = Tk()
root.configure(bg="#ffe6e6")
root.geometry("600x600")


def onclick(root=root):
    path = askopenfilename()
    equation, array_line, final_image, output = segment(path)

    new_img_for_gui = np.copy(final_image)
    for i in array_line:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(new_img_for_gui, str(i[0]), (i[3], i[4]), font, 2, (0, 0, 255), 5, cv2.LINE_AA)

    img = cv2.cvtColor(new_img_for_gui, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    im = im.resize((300, 200), Image.ANTIALIAS)
    im.save("ArtWrk.ppm", "ppm")
    tkimage = ImageTk.PhotoImage(im)
    myvar = Label(root, image=tkimage)
    myvar.image = tkimage
    myvar.place(x=115, y=150)
    # ------------Text Box-------------
    T = Text(root, height=20, width=50)
    T.place(x=90, y=380)
    quote = equation
    T.insert(END, quote)
    verilog(equation, output)


# Adding widgets to the root window
Label(root, text='Gates Analyzer', font=('Verdana', 15)).pack(side=TOP, pady=10)

# Creating a photoimage object to use image
photo = PhotoImage(file=r"D:/Nada/4th Elec/Second Term/Image Processing/cvc_logo.PNG")

# Resizing image to fit on button
photoimage = photo.subsample(9, 9)

Button(root, text='Click Me !', image=photoimage, compound=LEFT, command=onclick).pack(side=TOP)

root.mainloop()

# D:/Nada/4th Elec/Second Term/Image Processing/Project/Dataset/gates/draw_15.jpg




