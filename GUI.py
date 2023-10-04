from tkinter import *
from PIL import Image, ImageTk  
from Testbot import predictbot



def pp(a):
    global mylist
    mylist.insert(END, a)


def predict(val):
    print(val)
    
    output=predictbot(val)
    
    
    
    root.after(500, lambda : pp("Model loaded"))
    root.after(1700, lambda : pp("Text preprocessing"))
    root.after(2000, lambda : pp("Feature Extraction"))
    root.after(2500, lambda : pp("Prediction"))
    root.after(2800, lambda : pp("Result: "+output))
    root.after(3000, lambda : pp("============================"))
    root.after(3100, lambda :shrslt.config(text=output,fg="red"))
        
    
    
    
def userHome():
    global root, mylist,shrslt
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    image = Image.open("twbot.jpg")
    image = image.resize((1200, 700), Image.ANTIALIAS) 
    pic = ImageTk.PhotoImage(image)
    lbl_reg=Label(root,image=pic,anchor=CENTER)
    lbl_reg.place(x=0,y=0)
  
    #-----------------INFO TOP------------
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="CONTENT BASED BOT DETECTION",fg="white",bg="#000955",bd=10,anchor='w')
    lblinfo.place(x=400,y=20)
    
    
    lblinfo3 = Label(root, font=( 'aria' ,20 ),text="Enter Tweet ",fg="#000955",anchor='w')
    lblinfo3.place(x=780,y=310)
    E1 = Entry(root,width=30,font="veranda 20")
    E1.place(x=650,y=360)
    mylist = Listbox(root,width=50, height=20,bg="white")
    lblinfo4 = Label(root, font=( 'aria' ,16 ),text="Process ",fg="#000955",anchor='w')
    lblinfo4.place(x=180,y=270)

    mylist.place( x = 80, y = 300 )
    btntrn=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Detect", bg="red",command=lambda:predict(E1.get()))
    btntrn.place(x=800, y=420)
    # btnhlp=Button(root,padx=16,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Help?", bg="blue",command=lambda:predict(E1.get()))
    # btnhlp.place(x=50, y=450)
    rslt = Label(root, font=( 'aria' ,20, ),text="RESULT :",fg="black",bg="white",anchor=W)
    rslt.place(x=640,y=580)
    shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=780,y=580)

    def qexit():
        root.destroy()
     

    root.mainloop()


userHome()