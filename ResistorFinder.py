from tkinter import *
from tkinter.ttk import *
import tkinter.font as font
    
colors = ["black","brown","red","orange","yellow", "green","blue","violet","gray","white"]

class A(Frame):
    def __init__(self, a):
        Frame.__init__(self, a)
        self.grid()
		
        self.color = ["black","black","black","black"]
        
        self.create_elements()
    def create_elements(self):
 
        self.res = Canvas(self, width=300, height=100)
        self.res.config(bg="grey")
        self.res.create_rectangle((10,10,290,90), fill="#EDC9AF")
        self.res.grid(row=0,column=0,columnspan=3)

        Label(self, text = "First band:").grid(row=1, column=0)
        self.band1 = Canvas(self, width=200, height=50)
        self.draw_colors(self.band1)
        self.band1.grid(row=1,column=1, columnspan = 2)
        self.band1.bind("<Button 1>", self.b1)

        Label(self, text = "Second band:").grid(row=2, column=0)
        self.band2 = Canvas(self, width=200, height=50)
        self.draw_colors(self.band2)
        self.band2.grid(row=2,column=1, columnspan = 2)
        self.band2.bind("<Button 1>", self.b2)

        Label(self, text = "Third band:").grid(row=3, column=0)
        self.band3 = Canvas(self, width=200, height=50)
        self.draw_colors(self.band3)
        self.band3.grid(row=3,column=1, columnspan = 2)
        self.band3.bind("<Button 1>", self.b3)

        Label(self, text = "Fourth band:").grid(row=4, column=0)
        self.band4 = Canvas(self, width=200, height=50)
        self.band4.create_rectangle((0,0,100,50),fill="gold", outline="gold")
        self.band4.create_rectangle((100,0,200,50),fill="gray", outline="gray")
        self.band4.create_text((50, 25), text="+/- 5%")
        self.band4.create_text((150, 25), text="+/- 10%")
        self.band4.grid(row=4,column=1, columnspan = 2)
        self.band4.bind("<Button 1>", self.b4)

        self.result = Text(self, width=25, height=1)
        self.result.grid(row=5,column=1, columnspan=3)

        self.update()
    def calculate(self):
        val = str(colors.index(self.color[0]))+str(colors.index(self.color[1]))
        for i in range(colors.index(self.color[2])):
            val+="0"
        return val   
    
    def draw_colors(self, canv):
        for i in range(10):
            canv.create_rectangle((20*i,1,20+20*i,50),fill=colors[i], outline=colors[i])
    def update(self):
        for i in range(4):
            self.res.create_rectangle((60*i+40,10,60*i+70,90), fill=self.color[i])
        self.result.delete(0.0, END)
        self.result.insert(END, 'Value: '+self.calculate()+ " ohms")
        self.result.configure(bg='pink')
        
		
    def change(self, band, color):
        self.color[band-1] = color
        self.update()
    def b1(self, event):
        self.change(1, self.xToc(event.x))
    def b2(self, event):
        self.change(2, self.xToc(event.x))
    def b3(self, event):
        self.change(3, self.xToc(event.x))
    def b4(self, event):
        if event.x <= 100:
            self.change(4, "gold")
        else:
            self.change(4, "gray")
    def xToc(self, x):
        return colors[int(x/20)]
        
Finder = Tk()
Finder.title("Resistor Code Finder")
app = A(Finder)
Finder.mainloop()
