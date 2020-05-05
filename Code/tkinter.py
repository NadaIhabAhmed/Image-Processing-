

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
