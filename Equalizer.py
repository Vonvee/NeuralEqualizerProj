#Made by Fedor Irkhin
#2017 HSE

import numpy as np
import random
import math
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#Global functions
def LinearCanal(a, NC, j_s, noise):
    x = [0] * len(a)
    for i in range(NC):
        x[i] = a[i]

    # NL = 0
    for i in range(NC, len(a)):
        o = 0
        for j in range(NC):
            o = o + a[i - j] * j_s[NC - j - 1]

            x[i] = o
        x[i] = x[i] + np.random.normal(scale=noise)
    return x;


def NL1Canal(a, NC, j_s, noise):
    x = [0] * len(a)
    for i in range(NC):
        x[i] = a[i]

    # NL = 1
    for i in range(NC, len(a)):
        o = 0
        for j in range(NC):
            o = o + a[i - j] * j_s[NC - j - 1]
            x[i] = o
        x[i] = x[i] + x[i] * x[i] * 0.02 + x[i] * x[i] * x[i] * 0.01 + np.random.normal(scale=noise)
    return x;


#
def NL2Canal(a, NC, j_s, noise):
    x = [0] * len(a)
    for i in range(NC):
        x[i] = a[i]

    # NL = 2
    for i in range(NC, len(a)):
        o = 0
        for j in range(NC):
            o = o + a[i - j] * j_s[NC - j - 1]
            x[i] = o
        x[i] = x[i] + x[i] * x[i] * 0.2 + x[i] * x[i] * x[i] * 0.1 + np.random.normal(scale=noise)
    return x;


def predict_matrix(xtest, NC, clf):
    X_out = np.zeros(shape=(len(xtest),))
    for i in range(len(xtest)):
        X_out[i] = clf.predict([xtest[i]])
    return X_out


def to_matrix(a, NC, canal, j_s, noise):
    x = canal(a, NC, j_s, noise)
    X_tr = np.zeros(shape=((len(a) - (NC - 1)), NC))
    for i in range(NC - 1, len(a)):
        for j in range(NC):
            X_tr[i - (NC - 1), j] = x[i - j]
    y = a[:-(NC - 1)]
    return X_tr, y


from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import *
import fileinput
from sklearn.externals import joblib
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.neural_network import MLPRegressor


class Equalizer:
    # Varivables
    n = 100000
    NC = 20
    n_test = 5000+20
    noise = 0.01
    sigma = 1.0
    JCanal = (1,0.2,0.1)

    def get_test_data(self):
        a_test = np.zeros(shape=(self.n_test,))
        for i in range(self.n_test):
            a_test[i] = np.random.normal(scale=self.sigma)
        return a_test

    def __init__(self):
        #Initializing networks

        self.canal = NL2Canal

        root.minsize(width=1050, height=550)
        root.maxsize(width=1050, height=550)


        np.random.seed(seed=2)
        self.j_s = np.random.normal(size=self.NC)
        np.random.seed()
        #
        main_menu = Menu(root)
        root.config(menu=main_menu)

        self.test_data = self.get_test_data()

        try:
            self.MyClf = joblib.load('C:/Users/Fedor/Documents/MLP084r2.pkl')
            self.LinClf = joblib.load('C:/Users/Fedor/Documents/Lin084r2.pkl')
        except:
            self.mlp_learn()


        first_menu = Menu(main_menu)
        main_menu.add_cascade(label="File", menu=first_menu)
        first_menu.add_command(label="Open equalizer", command=self.open_my_equalizer)
        first_menu.add_command(label="New equalizer", command=self.new_eq_params)
        first_menu.add_command(label="Save equalizer", command=self.save_my_eq)
        first_menu.add_command(label="Exit", command=self.close_win)

        canal_menu = Menu(main_menu)
        main_menu.add_cascade(label="Canals", menu=canal_menu)
        canal_menu.add_command(label="Show canals", command=self.Plot_canals)
        canal_menu.add_command(label="Choose canal", command=self.choose_canal)

        signal_menu = Menu(main_menu)
        main_menu.add_cascade(label="Signals", menu=signal_menu)
        signal_menu.add_command(label="Open signal", command=self.open_signal)
        signal_menu.add_command(label="Generate signal", command=self.generate_signal)
        signal_menu.add_command(label="Save signal", command=self.save_signal)

        build_menu = Menu(main_menu)
        main_menu.add_cascade(label="Build", menu=build_menu)
        build_menu.add_command(label="Build Answer", command=self.build_answer)
        build_menu.add_command(label="Recover signal", command=self.recovery)

        second_menu = Menu(main_menu)
        main_menu.add_cascade(label="Help", menu=second_menu)
        #second_menu.add_command(label="Help")
        second_menu.add_command(label="About", command=self.about)
        # Label
        self.label_text = StringVar()
        self.label_text.set(" ")
        lis1 = Label(root, textvariable=self.label_text, font="Verdana 13")
        lis1.pack(side="top", pady=20)


    Neurons = (100,50)
    def mlp_learn(self):
        np.random.seed(seed=2)
        self.j_s = np.random.normal(size=self.NC)
        np.random.seed()

        # MLP Perc
        self.MyClf = MLPRegressor(activation='relu', solver='adam', hidden_layer_sizes=self.Neurons)
        a_tr = np.zeros(shape=(self.n,))
        for i in range(self.n):
            a_tr[i] = np.random.normal(scale=self.sigma)
        x_tr, y_tr = to_matrix(a_tr, self.NC, self.canal, self.j_s, self.noise)
        self.MyClf.fit(x_tr, y_tr)

        self.LinClf = LinearRegression()
        self.LinClf.fit(x_tr,y_tr)


    
    def set_params(self):
        #Set window parametrs 
        self.n = int(self.eN.get())
        self.NC = int(self.eNC.get())
        self.Sigma = float(self.eSigma.get())
        self.Noise = float(self.eNoise.get())
        self.Neurons = [int(x) for x in self.eNeurons.get().split(",")]
        self.JCanal = [int(x) for x in self.eNeurons.get().split(",")]
        self.mlp_learn()
        self.new_eq_win.destroy()

    def new_eq_params(self):
        #Make window with fields to new network params
        new_eq_win = Toplevel(root)
        new_eq_win.title("Set params")
        new_eq_win.maxsize(width=300, height=300)

        Label(new_eq_win, text="N").grid(row=0)
        Label(new_eq_win, text="NC").grid(row=1)
        Label(new_eq_win, text="Sigma").grid(row=2)
        Label(new_eq_win, text="Noise").grid(row=3)
        Label(new_eq_win, text="Neurons").grid(row=4)
        Label(new_eq_win, text="j").grid(row=5)

        self.eN = Entry(new_eq_win)
        self.eNC = Entry(new_eq_win)
        self.eSigma = Entry(new_eq_win)
        self.eNoise = Entry(new_eq_win)
        self.eNeurons = Entry(new_eq_win)
        self.eJ = Entry(new_eq_win)
        button = Button(new_eq_win, text='Start', command=self.set_params)

        self.eN.grid(row=0, column=1)
        self.eNC.grid(row=1, column=1)
        self.eSigma.grid(row=2, column=1)
        self.eNoise.grid(row=3, column=1)
        self.eNeurons.grid(row=4, column=1)
        self.eJ.grid(row=5, column=1)
        button.grid(row=6, columnspan=2)

        self.eN.delete(0, END)
        self.eN.insert(0, "100000")
        self.eNC.delete(0, END)
        self.eNC.insert(0, "20")
        self.eSigma.delete(0, END)
        self.eSigma.insert(0, "1.0")
        self.eNoise.delete(0, END)
        self.eNoise.insert(0, "0.01")
        self.eNeurons.delete(0, END)
        self.eNeurons.insert(0, "100,50")
        self.eJ.delete(0, END)
        self.eJ.insert(0, "1,0.1,0.2")

        self.new_eq_win = new_eq_win



    def set_canal(self):
        #Set canal to choosen one
        value = self.temp_canal.get()
        if (value == "LinearCanal"):
            self.canal = LinearCanal
        if (value == "NL1Canal"):
            self.canal = NL1Canal
        if (value == "NL2Canal"):
            self.canal = NL2Canal
        self.ch_canal_win.destroy()

    def open_signal(self):
        self.NC = 20
        types = [('Output Files', '*.out'), ('Txt Files', '*.txt')]
        op = askopenfilename(filetypes=types)
        try:

            f = open(op, "r")
            # LoadArray
            self.test_data = np.genfromtxt(f.name, dtype='float')
            self.n_test = len(self.test_data) + self.NC

        except:
            pass

    def save_signal(self):
        #Save signal as array
        types = [('Out files', '*.out'), ('Txt files', '*.txt')]
        save_as = asksaveasfilename(filetypes=types)
        try:
            f = open(save_as, "w")
            np.savetxt(f.name, self.test_data, delimiter=',')
            # f.write(letter)
            f.close()
        except:
            pass

    def choose_canal(self):
        #Choose canal out of 3
        self.ch_canal_win = Toplevel(root)
        self.ch_canal_win.title("Select canal")
        self.ch_canal_win.maxsize(width=250, height=250)

        Label(self.ch_canal_win, text="Canal").grid(row=0)

        self.temp_canal = StringVar(self.ch_canal_win)
        self.temp_canal.set("NL2Canal")  # default value

        self.canal_menu = OptionMenu(self.ch_canal_win, self.temp_canal, "LinearCanal", "NL1Canal", "NL2Canal")
        self.canal_menu.grid(row=0, column=1)

        cbutton = Button(self.ch_canal_win, text='Apply', command=self.set_canal)

        self.canal_menu.grid(row=0, column=1)
        cbutton.grid(columnspan=2)

    def set_n_test(self):
        #Set n_test variable
        self.n_test = int(self.eNtest.get()) + self.NC
        self.test_data = self.get_test_data()
        self.gen_signl_win.destroy()

    def generate_signal(self):
        #Generate new signal
        self.gen_signl_win = Toplevel(root)
        self.gen_signl_win.title("Generate new signal")
        self.gen_signl_win.maxsize(width=250, height=250)

        Label(self.gen_signl_win, text="N_test:").grid(row=0)

        self.eNtest = Entry(self.gen_signl_win)
        self.eNtest.grid(row=0, column=1)
        self.eNtest.delete(0, END)
        self.eNtest.insert(0, "5000")

        cbutton = Button(self.gen_signl_win, text='Apply', command=self.set_n_test)
        cbutton.grid(columnspan=2)

    def myplot(self, f, predict, real, text, linewidth, alpha, grid,isSecond):
        #Plot 2 arrays
        sbp1 = f.add_subplot(grid)

        sbp1.plot(real, 'b', linewidth=linewidth, alpha=alpha)
        if(isSecond == True):
            sbp1.plot(predict, 'r', linewidth=linewidth, alpha=alpha)

        sbp1.set_title(text)
        sbp1.set_xlabel('Iterations')
        sbp1.set_xlim(1, min(len(predict), len(real)) - 1)
        # sbp1.set_ylabel('A')

    was_drawn = False

    def build_answer(self):
        if (self.was_drawn == True):
            self.canvas.get_tk_widget().destroy()

        self.was_drawn = True
        f = Figure(figsize=(8, 6), dpi=100, facecolor='0.94')

        # LinTest
        x_test, y_test = to_matrix(self.test_data, self.NC, self.canal, self.j_s, self.noise)


        X_out1 = predict_matrix(x_test, self.NC, self.LinClf)
        self.myplot(f, X_out1, y_test, "Linear", 0.5, 0.5, 121,True)


        X_out2 = predict_matrix(x_test, self.NC, self.MyClf)
        self.myplot(f, X_out2, y_test, "NonLinear", 0.5, 0.5, 122,True)

        self.label_text.set("       LinearRgr R^2 score = " + "{:.8f}".format(r2_score(y_test, self.LinClf.predict(x_test)))
                            + "\t\t" + "      NonLinearRgr R^2 score = " + "{:.8f}".format(
            r2_score(y_test, self.MyClf.predict(x_test))))


        self.canvas = FigureCanvasTkAgg(f, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=LEFT, fill=BOTH, expand=1)


    def recovery(self):
        #Recovery given signal    
        types = [('Output Files', '*.out'), ('Txt Files', '*.txt')]
        op = askopenfilename(filetypes=types)
        try:

            f = open(op, "r")
            # LoadArray
            data = np.genfromtxt(f.name, dtype='float')
            
            
            if (self.was_drawn == True):
                self.canvas.get_tk_widget().destroy()

            self.was_drawn = True
            f = Figure(figsize=(8, 6), dpi=100, facecolor='0.94')


            X_mtrx = np.zeros(shape=((len(data) - (self.NC - 1)), self.NC))
            for i in range(self.NC - 1, len(data)):
                for j in range(self.NC):
                    X_mtrx[i - (self.NC - 1), j] = data[i - j]
            
            X_out1 = predict_matrix(X_mtrx, self.NC, self.LinClf)
            self.myplot(f, X_out1, X_out1, "Linear", 0.5, 0.5, 121,False)


            X_out2 = predict_matrix(X_mtrx, self.NC, self.MyClf)
            self.myplot(f, X_out2, X_out2, "NonLinear", 0.5, 0.5, 122,False)

            self.label_text.set("")


            self.canvas = FigureCanvasTkAgg(f, master=root)
            self.canvas.show()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            self.canvas._tkcanvas.pack(side=LEFT, fill=BOTH, expand=1)
        except:
            pass




    def Plot_canals(self):
        canals_win = Toplevel(root)
        canals_win.minsize(width=1150, height=600)
        canals_win.maxsize(width=1150, height=600)
        # FIGURE PLOT
        f = Figure(figsize=(12, 6), dpi=100)
        sbp1 = f.add_subplot(221)
        sbp2 = f.add_subplot(222)
        sbp3 = f.add_subplot(223)
        sbp4 = f.add_subplot(224)

        a = self.test_data
        n = len(self.test_data)
        f.tight_layout(pad=5, w_pad=3, h_pad=3)
        sbp1.plot(a, linewidth=0.25)
        sbp1.set_title('Pure signal')
        
        sbp1.set_xlim(1, n)
        # sbp1.set_ylabel('A')

        sbp2.plot(LinearCanal(a, self.NC, self.j_s, self.noise), linewidth=0.25)
        sbp2.set_title('Linear')
        sbp2.set_xlim(1, n)
        # sbp2.set_ylabel('A')

        sbp3.plot(NL1Canal(a, self.NC, self.j_s, self.noise), linewidth=0.25)
        sbp3.set_title('NL1Canal')
        sbp3.set_xlabel('Iterations')
        sbp3.set_xlim(1, n)
        # sbp3.set_ylabel('A')

        sbp4.plot(NL2Canal(a, self.NC, self.j_s, self.noise), linewidth=0.25)
        sbp4.set_title('NL2Canal')
        sbp4.set_xlabel('Iterations')
        sbp4.set_xlim(1, n)
        # sbp4.set_ylabel('A')


        canvas = FigureCanvasTkAgg(f, master=canals_win)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)


    def open_my_equalizer(self):
        types = [('PKL Files', '*.pkl')]
        op = askopenfilename(filetypes=types)
        try:
            f = open(op, "r")

            self.MyClf = joblib.load(f.name)
        except:
            pass

    def save_my_eq(self):
        types = [('PKL Files', '*.pkl')]
        save_as = asksaveasfilename(filetypes=types)
        try:
            f = open(save_as, "w")

            joblib.dump(self.MyClf, f.name)
            # f.write(letter)
            f.close()
        except:
            pass

    def close_win(self):
        if askyesno("Warning", "Are you shure you want to exit?"):
            root.destroy()

    def about(self):
        showinfo("Editor Authors", "Equalizer_v1.0 made by Fedor Irkhin (c)")


root = Tk()
root.title("Equalizer v1.0")

obj_menu = Equalizer()

root.mainloop()