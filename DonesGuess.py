import tkinter as tk
import random
window =tk.Tk()
window.config(bg='grey')

def randomNum():
    num=random.randint(1,50)
    return num

def buttonClick():
    name=numEntry.get()
    num1=randomNum()
    if name == str(num1):
    	com= str(num1)
    	result.delete(0,tk.END)
    	result.insert(0, com)
    	uno='You are Correct!'
    	guess.delete(0,tk.END)
    	guess.insert(0, uno)
    else:
    	com= str(num1)
    	result.delete(0,tk.END)
    	result.insert(0, com)
    	uno='You are Wrong!'
    	guess.delete(0,tk.END)
    	guess.insert(0, uno)

numLabel=tk.Label(window, text="Choose number from 1-50", justify='center', bg='gray')
numEntry=tk.Entry(window, justify='center', bg='pink',width=20)
button=tk.Button(window, text="Guess",command=buttonClick, width=16,justify='center',bg='orange',bd=2)
result=tk.Entry(window, justify='center', bg='yellow', width = 20)
guess=tk.Entry(window,bg='green', justify='center', width = 20)

numLabel.pack()
numEntry.pack(ipady=8)
button.pack()
result.pack(ipady=8)
guess.pack(ipady=8)
window.mainloop()
