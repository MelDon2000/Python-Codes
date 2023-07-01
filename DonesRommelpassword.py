import tkinter as tk
from tkinter import messagebox
import tkinter.font as font


window=tk.Tk()
window.title("Log in")
window.geometry("200x200")
window.config(bg="#BC544B")


def checking():
    
    username ="Yellow"
    password="Orange"
    enteredusername=usernameEntry.get()
    enteredPassword=passwordEntry.get()
    if password==enteredPassword and username==enteredusername:
        messagebox.showinfo("Log in System", "You Logged in")
    elif password==enteredPassword and username!=enteredusername:
        messagebox.showinfo("Log in System", "Incorrect Username!")
    elif password!=enteredPassword and username==enteredusername:
        messagebox.showinfo("Log in System", "Incorrect Password!")
    else:
        messagebox.showinfo("Log in System", "Incorrect!")


myFonts = font.Font(family='Times New Roman', size =15)
myFonts1 = font.Font(family='Calibri', size =15)

usernameLabel=tk.Label(window, text="Username:")
usernameLabel.config(bg="#BC544B", font=myFonts)


passwordLabel=tk.Label(window, text="Password:")
passwordLabel.config(bg="#BC544B",font=myFonts)
usernameEntry=tk.Entry(window,font =myFonts1)
passwordEntry=tk.Entry(window, show="*", font=myFonts1)

myFont = font.Font(family='Helvetica', size =15)

button=tk.Button(window, text="Enter", command=checking)
button.config(bg="pink", font=myFont)

usernameLabel.pack()
usernameEntry.pack()
passwordLabel.pack()
passwordEntry.pack()
button.pack()

window.mainloop()
