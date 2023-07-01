import tkinter as tk
window=tk.Tk()

def sliderUpdate(something):
    red=redSlider.get()
    green=greenSlider.get()
    blue=blueSlider.get()

    colour="#%02x%02x%02x" %( red, green, blue)
    a=str(colour)
    canvas.config(bg=colour)
    guess.config(text=a)

redSlider=tk.Scale(window, from_=0, to=255, command=sliderUpdate)
greenSlider=tk.Scale(window, from_=0, to=255, command=sliderUpdate)
blueSlider=tk.Scale(window, from_=0, to=255, command=sliderUpdate)

guess=tk.Label(window, text='', bg='green', justify='center', width = 10)
canvas=tk.Canvas(window, height=300, width=300)
redSlider.grid(row=1, column=1)
greenSlider.grid(row=1, column=2)
blueSlider.grid(row=1, column=3)
canvas.grid(row=2, column=1, columnspan=3)
guess.grid(row=3, column=2)
window.mainloop()
