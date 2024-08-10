import tkinter as tk
import tkinter.messagebox


# Initializes a popup window that returns user entry as str var
class SingleEntry(tk.Toplevel):
    def __init__(self, master=None, title=None, message=None, **kw):
        tk.Toplevel.__init__(self)
        self.master = master
        self.title(title)
        self.msg = message
        self.entry = tk.StringVar()
        self.load_form()
        self.transient()
        self.grab_set()
        self.wait_window()

    def load_form(self):
        row1 = tk.Frame(self)
        row2 = tk.Frame(self)

        lab = tk.Label(row1, width=15, text=self.msg, anchor='w')
        ent = tk.Entry(row1)
        button_confirm = tk.Button(row2, text="Confirm", command=lambda: self.set_entry(ent.get()))
        button_quit = tk.Button(row2, text="Quit", command=self._quit)

        row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT)
        ent.focus_set()
        button_confirm.pack(side=tk.RIGHT, padx=5, pady=5)
        button_quit.pack(side=tk.RIGHT, padx=5, pady=5)

        self.bind('<Return>', lambda event: self.set_entry(ent.get()))
        self.bind('<Escape>', lambda event: self._quit())


    def set_entry(self, entry):
        self.entry.set(entry)
        self.destroy()

    def _quit(self):
        self.destroy()


class DoubleSlider(tk.Toplevel):
    def __init__(self, master=None, title=None, range_min=0, range_max=100, tick_width=5, **kw):
        super().__init__(master=master, **kw)
        self.master = master
        self.title(title)
        self.range_min = range_min
        self.range_max = range_max
        self.min_val = tk.IntVar()
        self.max_val = tk.IntVar()
        self.tick_width = tick_width
        self.load_form()
        self.transient()
        self.grab_set()
        self.focus_set()
        self.wait_window()

    def load_form(self):
        row1 = tk.Frame(self)

        label_min = tk.Label(self, width=15, text='minimum', anchor='center')
        label_max = tk.Label(self, width=15, text='maximum', anchor='center')
        entry_min = tk.Scale(self,
                             from_=self.range_min,
                             to=self.range_max,
                             tickinterval=self.tick_width,
                             orient=tk.HORIZONTAL,
                             length=500)
        entry_max = tk.Scale(self,
                             from_=self.range_min,
                             to=self.range_max,
                             tickinterval=self.tick_width,
                             orient=tk.HORIZONTAL,
                             length=500)

        button_confirm = tk.Button(row1, text="Confirm",
                                   command=lambda: self.set_entry(entry_min.get(), entry_max.get()))
        button_quit = tk.Button(row1, text="Quit", command=self._quit)

        label_min.pack(side=tk.TOP, fill=tk.X, padx=5, pady=7)
        entry_min.pack(side=tk.TOP, fill=tk.X, padx=5, pady=7)
        label_max.pack(side=tk.TOP, fill=tk.X, padx=5, pady=7)
        entry_max.pack(side=tk.TOP, fill=tk.X, padx=5, pady=7)

        row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=7)
        button_confirm.pack(side=tk.RIGHT, padx=5, pady=5)
        button_quit.pack(side=tk.RIGHT, padx=5, pady=5)

        self.bind('<Return>', lambda event: self.set_entry(entry_min.get(), entry_max.get()))
        self.bind('<Escape>', lambda event: self._quit())

    def set_entry(self, entry_min, entry_max):
        if entry_min > entry_max:
            tk.messagebox.showwarning(title="value error", message="minimum is greater than max")
        else:
            self.min_val.set(entry_min)
            self.max_val.set(entry_max)
            self.destroy()

    def _quit(self):
        self.destroy()