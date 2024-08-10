import os
import sys

import pylab
import tkinter as tk
from tkinter import messagebox
import popup
import matplotlib.markers

import matplotlib.patches as patches
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as pyplot
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage import filters, util, measure, morphology
from skimage.filters import threshold_otsu

from commons import common, preprocessing_histogram_normalization


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.wm_title("skull stripping")
        self.frame = 0
        self.dataset_n = 0
        self.dataset_name = ''
        self.volume = None
        self.new_vol = None
        self.props = None
        self.current_prop = None
        self.vlsz = None

    def load_widgets(self):
        # fig, ax = pyplot.subplots()
        fig, _ = pyplot.subplots(1, 2, figsize=(6, 3))
        canvas = FigureCanvasTkAgg(fig, master=root)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        row_master = tk.Frame(self.master)
        row1 = tk.Frame(row_master)
        row2 = tk.Frame(row_master)
        row3 = tk.Frame(row_master)
        row4 = tk.Frame(row_master)
        row5 = tk.Frame(row_master)
        row6 = tk.Frame(row_master)

        btn_left = tk.Button(row2, text='< Frame', command=self.frame_prev)
        btn_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_right = tk.Button(row2, text='Frame >', command=self.frame_next)
        btn_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_dataset_prev = tk.Button(row1, text='< Dataset', command=self.dataset_prev)
        btn_dataset_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_dataset_next = tk.Button(row1, text='Dataset >', command=self.dataset_next)
        btn_dataset_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_segment = tk.Button(row3, text='save segment(S)', command=self.saveROI)
        btn_segment.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_autotake = tk.Button(row4, text="autosave segments", command=self.autosaveROI)
        btn_autotake.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_retake = tk.Button(row3, text='retake segment(R)', command=self.retake_prop)
        btn_retake.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_area = tk.Button(row4, text='set area threshold', command=self.set_prop_area)
        btn_area.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_wipe = tk.Button(row5, text='erase segment(Del)', command=self.wipeROI)
        btn_wipe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_overview = tk.Button(row5, text='review', command=self.overview)
        btn_overview.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_skip = tk.Button(row5, text='skip', command=self.skip_to_frame)
        btn_skip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        btn_exit = tk.Button(row6, text='Exit', command=self._quit, anchor='center')
        btn_exit.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5)

        row_master.pack(side=tk.RIGHT, fill=tk.BOTH, anchor='n')
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row3.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row4.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row5.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        row6.pack(side=tk.BOTTOM, fill=tk.X, anchor='s', padx=5, pady=5)

        self.master.bind('<Left>', lambda event: self.frame_prev())
        self.master.bind('<Right>', lambda event: self.frame_next())
        self.master.bind('q', lambda event: self.saveROI())
        self.master.bind('Q', lambda event: self.saveROI())
        self.master.bind('r', lambda event: self.retake_prop())
        self.master.bind('R', lambda event: self.retake_prop())
        self.master.bind('<Prior>', lambda event: self.dataset_prev())
        self.master.bind('<Next>', lambda event: self.dataset_next())
        self.master.bind('<Delete>', lambda event: self.wipeROI())
        # self.master.bind('<Escape>', lambda event: self.quit())

    def frame_next(self):
        self.frame += 1
        if self.frame == self.vlsz[2]:
            check = tk.messagebox.askokcancel(title="Last Image", message="Save and move on to next batch?",
                                              parent=self)
            if check:
                common.montageDisp(self.new_vol, image_name=self.dataset_name)
                common.save_result_as_png(self.new_vol,
                                          os.path.abspath(os.path.join(os.getcwd(), '..', "mask_regions")),
                                          self.dataset_name)
                self.dataset_n += 1
                self.new_vol, self.volume, self.dataset_name = load_dataset(
                    os.getcwd(),
                    self.dataset_n)
                self.vlsz = np.shape(self.volume)
                self.frame = 0
            else:
                self.frame -= 1
                return
        img = self.volume[:, :, self.frame]
        self.props = get_props(img)
        prop, self.props = refresh_prop(img, self.props)
        self.current_prop = prop
        displayROI(self.volume[:, :, self.frame], self.new_vol[:, :, self.frame], prop, self.frame)

    def frame_prev(self):
        if self.frame > 0:
            self.frame -= 1
            img = self.volume[:, :, self.frame]
            self.props = get_props(img)
            prop, self.props = refresh_prop(img, self.props)
            self.current_prop = prop
            displayROI(self.volume[:, :, self.frame], self.new_vol[:, :, self.frame], prop, self.frame)

    def dataset_next(self):
        check_save = tk.messagebox.askokcancel(title='save volume', message='save current mask selections?',
                                               parent=self)
        if check_save:
            common.save_result_as_png(self.new_vol, os.path.abspath(os.path.join(os.getcwd(), '..', "mask_regions")),
                                      self.dataset_name)
        self.dataset_n += 1
        self.new_vol, self.volume, self.dataset_name = load_dataset(
            os.getcwd(),
            self.dataset_n)
        self.vlsz = np.shape(self.volume)
        self.frame = 0

        img = self.volume[:, :, self.frame]
        self.props = get_props(img)
        prop, self.props = refresh_prop(img, self.props)
        self.current_prop = prop
        displayROI(self.volume[:, :, self.frame], self.new_vol[:, :, self.frame], prop, self.frame)

    def dataset_prev(self):
        check_save = tk.messagebox.askokcancel(title='save volume', message='save current mask selections?',
                                               parent=self)
        if check_save:
            common.save_result_as_png(self.new_vol, os.path.abspath(os.path.join(os.getcwd(), '..', "mask_regions")),
                                      self.dataset_name)
        self.dataset_n -= 1
        self.new_vol, self.volume, self.dataset_name = load_dataset(
            os.getcwd(),
            self.dataset_n)
        self.vlsz = np.shape(self.volume)
        self.frame = 0

        img = self.volume[:, :, self.frame]
        self.props = get_props(img)
        prop, self.props = refresh_prop(img, self.props)
        self.current_prop = prop
        displayROI(self.volume[:, :, self.frame], self.new_vol[:, :, self.frame], prop, self.frame)

    def saveROI(self):
        self.new_vol = copyROI(self.new_vol, self.current_prop, self.frame)
        img = self.volume[:, :, self.frame]
        displayROI(img, self.new_vol[:, :, self.frame], self.current_prop, self.frame)

        # common.montageDisp(self.new_vol, image_name='result_mask')

    def autosaveROI(self):
        new_window = popup.SingleEntry(master=self.master, title="autosave", message="n of frames to skip")
        entry = new_window.entry.get()
        if entry == '' or int(entry) == 0:
            return

        if int(entry) > (self.vlsz[2] - self.frame - 1):
            tk.messagebox.showwarning("index error",
                                      "entry %d exceeds number of remaining frames %d" % (int(entry),
                                                                                          (self.vlsz[
                                                                                               2] - self.frame)))
            return
        for _count in range(int(entry)):
            self.new_vol = copyROI(self.new_vol, self.current_prop, self.frame)
            self.frame += 1
            img = self.volume[:, :, self.frame]
            self.props = get_props(img)
            prop, self.props = refresh_prop(img, self.props)
            self.current_prop = prop

        displayROI(img, self.new_vol[:, :, self.frame], self.current_prop, self.frame)

    def set_prop_area(self):
        min_area = self.props[-1].area
        max_area = self.props[0].area
        new_window = popup.DoubleSlider(master=self.master, title='set area', range_min=min_area, range_max=max_area,
                                        tick_width=(max_area - min_area) / 15)

        img = self.volume[:, :, self.frame]
        self.props = get_props(img)
        prop, self.props = refresh_prop(img, self.props, thres_area_min=new_window.min_val.get(),
                                        thres_area_max=new_window.max_val.get())
        self.current_prop = prop

        displayROI(img, self.new_vol[:, :, self.frame], self.current_prop, self.frame)

    def skip_to_frame(self):
        new_window = popup.SingleEntry(title="skip", message="Skip to frame", master=self.master)
        entry = new_window.entry.get()
        if entry == '':
            return
        else:
            backup_frame = self.frame
            self.frame = int(entry)
            try:
                img = self.volume[:, :, self.frame]
            except IndexError:
                tk.messagebox.showwarning(title="index error", message="index out of range")
                self.frame = backup_frame
                return

            self.props = get_props(img)
            prop, self.props = refresh_prop(img, self.props)
            self.current_prop = prop
            displayROI(img, self.new_vol[:, :, self.frame], self.current_prop, self.frame)

    def wipeROI(self):
        # binary_image = self.new_vol[:, :, self.frame]
        # binary_image[self.current_prop.bbox[0]:self.current_prop.bbox[2],
        # self.current_prop.bbox[1]:self.current_prop.bbox[3]] = binary_image[
        #                                                        self.current_prop.bbox[0]:self.current_prop.bbox[2],
        #                                                        self.current_prop.bbox[1]:self.current_prop.bbox[3]
        #                                                        ] - np.multiply(self.current_prop.filled_image,
        #                                                                        255)
        x, y = self.current_prop.coords.T
        assert len(x) == len(y)
        for counter in range(len(x)):
            self.new_vol[x[counter], y[counter], self.frame] = 0
        # self.new_vol[:, :, self.frame] = binary_image

        img = self.volume[:, :, self.frame]

        displayROI(img, self.new_vol[:, :, self.frame], self.current_prop, self.frame)

    def retake_prop(self):
        img = self.volume[:, :, self.frame]
        prop, self.props = refresh_prop(img, self.props)
        self.current_prop = prop

        displayROI(img, self.new_vol[:, :, self.frame], prop, self.frame)

    def overview(self):
        common.montageDisp(self.new_vol,
                           image_name=self.dataset_name)

    def _quit(self):
        self.master.quit()  # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


# def copyROI(new_vol, prop, frame):
#     assert new_vol[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3], frame].shape == prop.filled_image.shape
#     new_vol[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3], frame] = new_vol[prop.bbox[0]:prop.bbox[2],
#                                                                            prop.bbox[1]:prop.bbox[3],
#                                                                            frame] + np.multiply(prop.filled_image, 255)
#
#     return new_vol


def copyROI(new_vol, prop, frame):
    x, y = prop.coords.T
    assert len(x) == len(y)
    for counter in range(len(x)):
        new_vol[x[counter], y[counter], frame] = 255

    return new_vol


def displayROI(img, binary_img, prop, frame=0):
    x = prop.bbox[1]
    y = prop.bbox[0]
    w = prop.bbox[3] - prop.bbox[1]
    h = prop.bbox[2] - prop.bbox[0]

    pyplot.clf()
    pyplot.title(str(frame))
    thresh = threshold_otsu(img)
    binary = img > thresh
    ax1 = pyplot.subplot(1, 2, 1, title=frame)
    ax2 = pyplot.subplot(1, 2, 2, title='mask')

    ax1.imshow(binary, cmap=pylab.cm.gist_gray)
    ax1.axis('off')
    # ax1.add_patch(patches.Rectangle((x, y),
    #                                 w,
    #                                 h,
    #                                 alpha=1,
    #                                 color='b',
    #                                 linewidth=2.5,
    #                                 fill=False))
    x, y = prop.coords.T
    ax1.scatter(y, x, c='#001eff', s=(72. / pyplot.gcf().dpi) ** 2, edgecolor="", alpha=1)

    ax2.imshow(binary_img, cmap=pylab.cm.gist_gray)
    ax2.axis('off')

    pyplot.gcf().canvas.draw()


def get_props(img):
    thresh = threshold_otsu(img)
    binary = img > thresh

    labels = measure.label(binary)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    props = measure.regionprops(labels)
    props.sort(key=lambda p: p.area, reverse=True)

    return props


def refresh_prop(image, props, thres_area_min=10, thres_area_max=5000000):
    for region in props:
        if region.area > thres_area_max:
            continue
        if region.area < thres_area_min:
            continue
        if 0 in region.coords[:, 0]:
            continue
        if 0 in region.coords[:, 1]:
            continue
        if (image.shape[0] - 1) in region.coords[:, 0]:
            continue
        if (image.shape[1] - 1) in region.coords[:, 1]:
            continue

        return region, props[props.index(region) + 1:]


def sortkey(e):
    ecomps = e.split('_')
    return int(ecomps[-1][:-4])


### Loads images in specific dataset
def load_dataset(root_path, dataset_n):
    subdirs = os.listdir(root_path)  # subdirs = ['DUM_0001', 'DUM_0002', ...]
    try:
        dataset = subdirs[dataset_n]
    except IndexError as error:
        print(error)
        tk.messagebox.showwarning(title='dataset index error', message='index out of range')
        sys.exit()

    full_path = os.path.join(root_path, dataset)
    imgs_ls = os.listdir(full_path)  # imgs_ls = ['DUM_0001_0', 'DUM_0001_1', ...]
    imgs_ls.sort(key=sortkey)
    img_path_ls = list()

    for img in imgs_ls:
        img_path_ls.append(os.path.join(full_path, img))

    raw_img_batch = common.read_raw_png(img_path_ls)
    print("### Reading %s ###" % dataset)

    # For datasets needing binarization
    data_binarized = preprocessing_histogram_normalization.preprocessing3D(raw_img_batch)
    print("### Binarization finished ###")

    data_shape = np.shape(raw_img_batch)
    new_vol = np.zeros(data_shape, np.uint8)
    # data_binarized = util.img_as_ubyte(raw_img_batch / np.max(raw_img_batch))
    # For datasets needing binarization
    # data_binarized = util.img_as_ubyte(data_binarized / np.max(data_binarized))

    common.montageDisp(data_binarized, 0, "./orig/", dataset, 'Binarized')

    return new_vol, data_binarized, dataset


if __name__ == '__main__':
    cwd = 'D:\\01_dataset\\data_CT\\disease_orig_png'  # Also change results save directory
    os.chdir(cwd)
    root = tk.Tk()
    app = Application(master=root)
    folder_name = input('folder name: ')
    subdirs = os.listdir(os.getcwd())
    app.dataset_n = subdirs.index(folder_name)
    app.new_vol, app.volume, app.dataset_name = load_dataset(os.getcwd(), app.dataset_n)
    app.vlsz = np.shape(app.volume)
    image = app.volume[:, :, app.frame]
    app.props = get_props(image)
    prop, app.props = refresh_prop(image, app.props)
    app.current_prop = prop
    app.load_widgets()
    displayROI(image, app.new_vol[:, :, app.frame], prop, app.frame)
    tk.mainloop()
