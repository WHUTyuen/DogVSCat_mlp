
from MyNet import NeuralNet
import torch
from PIL import Image
import numpy as np
import tkinter as tk

if __name__ == '__main__':

    net = torch.load("models/net.pth")
    img = Image.open("testimg/dog3.jpg")
    img = img.resize((100,100))
    img = np.array(img)
    img = torch.tensor(img,dtype=torch.float32)
    img = img/255 - 0.5

    img = torch.unsqueeze(img,0)
    out = net(img.cuda())
    print(out.detach().cpu().numpy())
    # print(out)
    res = "猫" if out.argmax(1).detach().cpu().numpy() == 0 else "狗"

    # GUI 框架
    win = tk.Tk()
    width , height = 300,50
    win.title("AI的预测结果:")
    label = tk.Label(win, text = "这是一只{},\n 预测率为:{}".format(res, str(round(out.detach().cpu().numpy().max() * 100,2)) + "%"))
    label.pack()

    align = "%dx%d+%d+%d" %(width,height, (win.winfo_screenwidth() - width)/2, (win.winfo_screenheight()- height) / 2 )
    win.geometry(align)
    win.mainloop()

