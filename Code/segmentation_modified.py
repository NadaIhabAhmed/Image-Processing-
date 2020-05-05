# -----------------------Imports-----------------------
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import imutils
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

# ------------------------------------------------------------------------
# ------------------------------COMPONENTS SEGMENTATION--------------------
def segment(path):
    # call the saved model
    pipeline = call_model()
    # Read Image
    img = cv2.imread(path)
    empty = (np.copy(img)) * 0
    white = (np.copy(img)) * 0 + 255
    original = (np.copy(img))
    lines_img = (np.copy(img))

    And = []
    Or = []
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

        # a new figure for every (plt)plot, (necessary nefore each plt.imshow(img) to be able to show many images)
        #fig = plt.figure()

        empty[y - 20: y + h + 20, x - 20: x + w + 20] = original[y - 20: y + h + 20, x - 20: x + w + 20]
        lines_img[y - 20: y + h + 20, x - 20: x + w + 20] = white[y - 20: y + h + 20, x - 20: x + w + 20]

        new_img_2 = new_img_2[y - 20: y + h + 20,
                    x - 20: x + w + 20]  # crop the image in the place of the rectabgle contour

        # --------------------------------
        # prediction part (ده الجزء بتاع كود مارينا)
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
            And.append([x, y, w, h])
        else:
            array_gates.append(['O', id_g, x1_g, y1_g, x2_g, y2_g, x3_g, y3_g, x4_g, y4_g])
            Or.append([x, y, w, h])

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

        plt.imshow(new_img_2)
        plt.text(10, 180, 'Predicted: %s' % preds[0], color='k', backgroundcolor='red', alpha=0.8)
        plt.show()
        # نهاية جزء مارينا
        # -------------------------------
        center = (x, y)
        print("center -- >", center)
        num_rect = num_rect + 1

    print("Number of rectangles = " + str(num_rect))
    print("Mean area = ", mean_area)
    # print(contours)
    print('Numbers of contours found=' + str(len(contours)))

    # use -1 as the 3rd parameter to draw all the contours
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    plt.imshow(original)
    plt.show()
    plt.imshow(empty)
    plt.show()
    plt.imshow(original - empty)
    plt.show()
    plt.imshow(lines_img)
    plt.show()

    # Preprocessing part for lines
    ret, thresh_1 = cv2.threshold(lines_img, 70, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)  # note this is a horizontal kernel
    e_im = cv2.erode(thresh_1, kernel, iterations=3)
    d_im = cv2.dilate(e_im, kernel, iterations=2)
    plt.imshow(d_im)
    plt.show()
    '''
    print("And\n")
    print(And)
    print("ــــــــــــــــــــــــــــــــــــــــــــــــ")
    print("Or\n")
    print(Or)
    '''
    # -------------------------------END OF COMPONENTS SEGMENTATION-------------------
    # --------------------------------------------------------------------------------
    # ---------------------------LINES COORDINATES------------------------------------

    original = (np.copy(d_im))
    array_line = []
    id_l = 0
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
        '''cv2.circle(original, extTop, 8, (255, 0, 0), -1)
        cv2.circle(original, extBot, 8, (255, 255, 0), -1)'''
        x1_l = extRight[0]
        x2_l = extLeft[0]
        y1_l = extRight[1]
        y2_l = extLeft[1]
        id_l = id_l + 1
        array_line.append([id_l, x1_l, y1_l, x2_l, y2_l])

    plt.imshow(original)
    plt.show()
    # -------------END of End Points----------------
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # TESTING PART
    print(" ----------- GATES COORDINATES -----------")
    for coor_gate in array_gates:
        print(coor_gate)
    print(" ----------- LINES COORDINATES -----------")
    for coor_line in array_line:
        print(coor_line)
    # ---------------------------------------------------------------
    check_line_connect_gate(array_gates, array_line)


# ----------------------------------------------------------------
def check_line_connect_gate(array_gates, array_line):
    array_of_connections = []  # [line_id, start_gate_name, start_gate_id, end_gate_name, end_gate_id]
    for line in array_line:
        for gate in array_gates:
            # right part start point
            if abs(gate[2] - line[1]) < 150 and abs(gate[3] - line[2]) < 150 or \
                    abs(gate[4] - line[1]) < 150 and abs(gate[5] - line[2]) < 150:
                array_of_connections.append([line[0], gate[0], gate[1]])
            # left part end point
            if abs(gate[6] - line[3]) < 60 and abs(gate[7] - line[4]) < 60 or \
                    abs(gate[8] - line[3]) < 60 and abs(gate[7] - line[4]) < 60:
                array_of_connections[line[0] - 1].append(gate[0], gate[1])
    print("-------------------------------------------------------")
    print("ARRAY OF CONNECTIONS --NEW PART--")
    print(array_of_connections)



# ----------------------------------------------------------------

def main():
    exit = ""
    while exit != "0":
        path = input("Enter Images path: ")
        segment(path)
        exit = input("do you want to exit: ")


main()


'''
def onclick():
    path = askopenfilename()
    segment(path)
    
    im = Image.open(path)
    im = im.resize((300, 200), Image.ANTIALIAS)
    im.save("ArtWrk.ppm", "ppm")
    tkimage = ImageTk.PhotoImage(im)
    myvar = Label(root, image=tkimage)
    myvar.image = tkimage
    myvar.place(x=115, y=150)
    


# creating tkinter window
root = Tk()
root.configure(bg="#ffe6e6")
root.geometry("600x600")
# Adding widgets to the root window
Label(root, text='Gates Analyzer', font=(
    'Verdana', 15)).pack(side=TOP, pady=10)

# Creating a photoimage object to use image
photo = PhotoImage(file=r"D:/Nada/4th Elec/Second Term/Image Processing/cvc_logo.PNG")

# Resizing image to fit on button
photoimage = photo.subsample(9, 9)

Button(root, text='Click Me !', image=photoimage,
       compound=LEFT, command=onclick).pack(side=TOP)

T = Text(root, height=20, width=50)
T.place(x=90, y=380)
quote = ""
T.insert(END, quote)

mainloop()
'''


# D:/Nada/4th Elec/Second Term/Image Processing/Project/Dataset/gates/draw_15.jpg




