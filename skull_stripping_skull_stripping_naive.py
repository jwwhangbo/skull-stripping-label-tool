
import tkinter as tk
from tkinter import messagebox

import numpy as np
from skimage import filters, util, measure, morphology, exposure
import os, sys, cv2, pylab
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.getcwd(), '../common'))
from commons import filtering, common


class Application(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.master=master
		self.frameCount=0
		self.propCount=0
		self.answer=False
		self.chngFrame=False
		self.roiOpen=False
		self.wipeprop=False
		self.master.title("app.py")
		self.pack()
		self.LoadWidgets()
		# self.main()

	def LoadWidgets(self):
		btn_left = tk.Button(self, text='<', command=self.FramePrev)
		btn_left.grid(row = 0, column = 0, sticky = 'ew', padx=2, pady=2)
		self.master.bind('<Left>', lambda event : self.FramePrev())

		btn_right = tk.Button(self, text='>', command=self.FrameNext)
		btn_right.grid(row = 0, column = 1, sticky = 'ew', padx=2, pady=2)
		self.master.bind('<Right>', lambda event : self.FrameNext())

		btn_segment = tk.Button(self, text='save segment(S)', command=self.saveSegment)
		btn_segment.grid(row = 1, column = 0, columnspan = 2, sticky = 'ew', padx=2, pady=2)
		self.master.bind('s', lambda event : self.saveSegment())

		btn_retake = tk.Button(self, text='retake segment(R)', command=self.retakeSegment)
		btn_retake.grid(row = 2, column = 0, columnspan = 2, sticky = 'ew', padx=2, pady=2)
		self.master.bind('r', lambda event : self.retakeSegment())

		btn_wipe = tk.Button(self, text='erase segment(Del)', command=self.wipeSegment)
		btn_wipe.grid(row=3, column=0, columnspan=2,sticky='ew',padx=2,pady=2)
		self.master.bind('<Delete>', lambda event: self.wipeSegment())

		btn_exit = tk.Button(self, text='Exit(Esc)', command=self.exit)
		btn_exit.grid(row = 4, column = 0, columnspan = 2, sticky = 'ew', padx=2, pady=2)
		self.master.bind('<Escape>', lambda event : self.exit())

	def FrameNext(self):
		# self.frameCount += 1
		self.roiOpen = False
		self.chngFrame = True

	def FramePrev(self):
		if (self.frameCount == 0):
			pass
		else: 
			self.frameCount -= 2
			self.chngFrame = True
			self.roiOpen = False

	def saveSegment(self):
		self.answer = True

	def retakeSegment(self): 
		self.propCount += 1
		self.roiOpen = False

	def wipeSegment(self):
		self.wipeprop = True

	def exit(self):
		plt.close('all')
		cv2.destroyAllWindows()
		self.master.destroy()
		sys.exit()

	def volGrad(self, vol, method=ndimage.sobel):
		# volume gradient magnitude
		vol = filtering.unsharp_filter(vol)
		Gmag = ndimage.generic_gradient_magnitude(vol, method)
		Gmag = exposure.adjust_gamma(Gmag,0.01)
		Gmag_thres = filters.threshold_otsu(Gmag)
		Gmag_ret = Gmag < Gmag_thres
		return Gmag_ret

	def preprocessing2D(self, img, disk_sz=1):
		# Get image entropy (input image should be in a range [-1 1])
		max_val = img.max()
		new_img = filters.rank.entropy(img / max_val, morphology.disk(disk_sz))

		# normalization
		max_entropy = new_img.max()
		new_img = exposure.equalize_adapthist(new_img / max_entropy)

		min_entropy = new_img.min()
		max_entropy = new_img.max()
		#     print(min_entropy, max_entropy)
		new_max = 65535
		new_img = (new_img - min_entropy) * new_max / (max_entropy - min_entropy)
		new_img[new_img > new_max] = new_max
		new_img[new_img < 0] = 0
		inv_new_img = new_max - new_img

		# data type conversion from float64 to uint32
		new_img.astype(np.float32)
		inv_new_img.astype(np.float32)

		#     plt.imshow(new_img)
		#     plt.title('filtered image')
		#     plt.show()

		#     print(inv_new_img)
		#     plt.imshow(inv_new_img)
		#     plt.title('inverse image')
		#     plt.show()
		Gmag_thres = filters.threshold_otsu(new_img)
		new_img_ret = new_img > Gmag_thres
		Gmag_thres = filters.threshold_otsu(inv_new_img)
		inv_new_img_ret = inv_new_img > Gmag_thres
		return new_img_ret, inv_new_img_ret

	def preprocessing3D(self, vol):
		new_vol = np.zeros(vol.shape)
		inv_vol = np.zeros(vol.shape)
		for frame in range(vol.shape[2]):
			#         print('frame: '+str(frame))
			new_vol[:, :, frame], inv_vol[:, :, frame] = self.preprocessing2D(vol[:, :, frame])
		return new_vol, inv_vol


# ### Filtering

# In[4]:


	def dispROI(self, img, prop, frame = 0):
		x = prop.bbox[1]
		y = prop.bbox[0]
		w = prop.bbox[3]-prop.bbox[1]
		h = prop.bbox[2]-prop.bbox[0]
		fig,ax = plt.subplots()
		plt.imshow(img, cmap= pylab.cm.gist_gray)
		currentAxis = plt.gca()
		currentAxis.add_patch(patches.Rectangle((x,y), w,h, alpha=1, color='b', linewidth = 2.5, fill= False))
		plt.title(str(frame))
		plt.show()

	def copyROI(self, new_vol, img, prop, frame, folder=None):
		if (not self.roiOpen):
			self.dispROI(img, prop, frame)
			self.roiOpen = True
		if self.answer == True:
			for rows in range(prop.bbox[0],prop.bbox[2]):
				for cols in range(prop.bbox[1],prop.bbox[3]):
					if prop.filled_image[rows-prop.bbox[0],cols-prop.bbox[1]] == 1:
						new_vol[rows,cols,frame] = np.multiply(prop.filled_image[rows-prop.bbox[0], cols-prop.bbox[1]], 255)
					else:
						pass

			self.roiOpen = False
			self.answer = False
			common.montageDisp(new_vol, 0, "./result_mask/", folder, 'selected regions')

		if self.wipeprop == True:
			for rows in range(prop.bbox[0],prop.bbox[2]):
				for cols in range(prop.bbox[1],prop.bbox[3]):
					new_vol[rows,cols,frame] = np.multiply(prop.filled_image[rows-prop.bbox[0], cols-prop.bbox[1]], 0)
			
			self.roiOpen = False
			self.wipeprop = False
			common.montageDisp(new_vol, 0, "./result_mask/", folder, 'selected regions')
		
		else:
			self.update_idletasks()
			self.update()
			# pass
			
		return new_vol

	def filtProps(self, orig, vol, fold = None, thres_area=50, thres_weight = 0.3):
		self.frameCount = 0
		self.propCount = 0

		sz = np.shape(vol)
		new_vol = np.zeros(sz, np.uint8)
		vol = util.img_as_ubyte(vol/np.max(vol))
		
		while self.frameCount <= sz[2]:

			if (self.frameCount == sz[2]):
				check = tk.messagebox.askokcancel(title="Note", message="Move on to next batch?", parent=self)
				if check is True:
					break
				else:
					self.frameCount -= 1
					continue

			img = vol[:,:,self.frameCount]
			
			regions = measure.label(img)        
			props = measure.regionprops(regions, img)
			props = sorted(props, key=lambda p: p.area, reverse=True)
			self.propCount = 0
			# for prop in props:
			#     if  prop.coords.all() != 0 and prop.coords[:,0].all() != sz[0]-1 and prop.coords[:,1].all() != sz[1]-1: 
			#         if prop.filled_area > thres_area and passing == False:
			#             new_vol, passing = copyROI(new_vol, img, prop, frame, asking, passing)
			while (self.propCount < len(props)):
				if (self.chngFrame == True): 
					self.chngFrame = False
					break
				prop = props[self.propCount]
				# print(np.shape(prop))
				# sys.exit(0)
				if prop.coords.all() != 0 and prop.coords[:,0].all() != sz[0]-1 and prop.coords[:,1].all() != sz[1]-1:
					if prop.filled_area > thres_area:
						new_vol = self.copyROI(new_vol, img, prop, self.frameCount, fold)
					else:
						self.propCount += 1
				else:
					self.propCount += 1
			# common.montageDisp(new_vol, 0, "./result_mask/",folder, 'selected regions')
			# back = input('you wanna do again for this image?(y/n)')
			# if 'y' in back :
			#     frame = frame
			# else : 
			#     frame = frame +1
			countWarning = tk.messagebox.askokcancel(title="Note", 
													   message=" Move on to next image?",
													   parent=self)
			if countWarning is True:
				self.frameCount += 1
			else:
				pass
							
		return new_vol

root = tk.Tk()
app = Application(master=root)


def main():
	folder_root = os.path.join(os.getcwd(), '../disease')
	# folder_root = '/data2/KP/JBS-05K/CN/CT/'
	folder_ls = os.listdir(folder_root)
	# res_save = 1
	# tmp_fp = open('progress.txt', 'r')
	# progress = tmp_fp.read()
	# tmp_fp.close()
	# print("continuing from %s" % (progress))
	# conversion for the dicom file[0:1]
	for folder in folder_ls[int(progress):]:
		# pn_name = folder #folder.split('/')[5]
		# print(str(folder_ls.index(folder)) + '/' + str(len(folder_ls)))
		fold = os.path.join(folder_root,folder)
		path = common.check_tags(fold)
		if path == None:
			continue
		else:
			if len(path) > 0:
				data_ct, dcm_info = common.read_dicom(path)  
				# print(np.shape(data_ct)) # (row, column, frame)
				common.montageDisp(data_ct, res_save, "./orig/", folder, 'original images')
				print("### Finished loading original image ###")
				# data_binarized = app.volGrad(data_ct)
				_, data_binarized = app.preprocessing3D(data_ct)
				print("### Binarized image on %s ###" % (folder))
				common.montageDisp(data_binarized, res_save, "./result_mask/", folder, 'binarized images')
				data_result = app.filtProps(data_ct, data_binarized, pn_name)  
				common.montageDisp(data_result, res_save, "./result_mask/", folder, 'selected regions')
				common.saveResult(data_result, "./result_mask/", folder)
				print("### Saving selected regions for %s ###" % (folder))
				tmp_fp = open('progress.txt', 'w')
				tmp_fp.write(str(folder_ls.index(folder)+1))
				tmp_fp.close()

if (__name__ == '__main__'):
	main()