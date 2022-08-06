import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm
from scipy.signal import savgol_filter, argrelmin, argrelmax
import random as r
import os

## Written by Matteo Fulghieri with help from Elizabeth Ryland, Natalia Powers-Riggs, and Trevor Wiechmann

## This code is meant to be a tester for an interference pattern should the Markov Chain Monte Carlo
## fail to give a reasonable result. In this code, you get to see the data corrections (the smoothing and 
## normalizing process). You can then manually choose values to try for fitting the data curves. I urge you
## to first try the MCMC/maunal results, or to just spam 0 + <Ent> to see the data. It also allows you to continually
## tweak your guess, if you would like to see new guesses over and over.  


def init():
## This function simply reads the polymer as an input,
## and that's it. It's the first user input when 
## code is executed. The polymer type is used for the
## index of refraction and titling purposes
	
	## Ask for polymer type
	polymer = input("Please type polymer (e.g. PVC, PS, PMMA): ")
	
	## Verify that code recognizes the polymer
	if polymer == "PVC":
		print("Polyvinyl Chloride")
		return polymer
	if polymer == "PS":
		print("Polystyrene")
		return polymer
	if polymer == "PET":
		print("Polyethylene terephthalate")
		print("Not yet supported")
		return False
	if polymer == "PP":
		print("Polypropylene")
		print("Not yet supported")
		return False
	if polymer == "PTFE":
		print("Polytetrafluoroethylene")
		print("Not yet supported")
		return False
	if polymer == "PMMA":
		print("Poly(methyl methacrylate)")
		print("Not yet supported")
		return False
	else: 
		print("Polymer not recognized")
		return False


def read_file(file_name):
## This function reads in the .csv file after the file name has
## been given. It then performs data correction on that file. 
## Data correction includes subtraction of blank scan, interpretation 
## of non-data from .csv, and normalization of attenuation.

	## Change into the data directory
	os.chdir("../Data")
	
	## Obtain headers from file
	header = np.genfromtxt("{0}".format(file_name), delimiter=',', dtype=str, max_rows=2)
	
	## Get number of scans from the length of the first row, minus the extra comma at the end
	row1 = header[0,:]
	num_of_scans = int((row1.shape[0] - 1) / 2)
	print("This file contains {0} scans (including blank).".format(num_of_scans))

	## Generate data
	#### The skip_footer can throw an error; 27-29 usually works, 
	#### but it depends on the scan settings
	data = np.genfromtxt("{0}".format(file_name), delimiter=",", dtype=float, skip_header=2, skip_footer=29*num_of_scans) 

	## Remove the trailing column of Nan's
	#### This is a result of the trailing commas
	data = np.delete(data,num_of_scans * 2,1)
	
	## Subtract out the noise from the blank scan
	#### Data is stored in every even row (or I guess odd since 0-indexed)
	data[:,1::2] -= data[:,1,None]

	## Remove the first two columns, since it is just the blank
	for i in range(2):
		data = np.delete(data,0,1)
		header = np.delete(header,0,1)
	
	## Remove blank scan from the count
	num_of_scans -= 1

	### Subtract out minimum to have data go to zero (i.e. wavelength independent attenuation)
	#temp = data[300:,:]
	#for k in range(num_of_scans):
	#	data[:,2*k+1] -= np.amin(temp[:,2*k+1])
	
	## Normalize data (Rayleigh scattering)
	data = normalize(data,num_of_scans)	

	return header, data, num_of_scans


def func(X,d):
## This is the model function to fit to. The model comes
## from literature. 

	x,n = X
	r = ( (n - 1) / (n + 1) ) ** 2
	return -np.log10((1 - r)**2 / (1 + r**2 - 2*r*np.cos(4*np.pi*n*d/x))) 


def inp():
## Input function to determine if input is valid
	temp = input()
	test = temp.replace(".","")
	if (not test.isdigit()):
		print("Try Again")
		inp()
	else:
		return temp




def plot_spectra(header, data, num_of_scans, file_name, polymer):
## This function plots the data as a spectrum, the fits, and the 
## output parameters. It also saves the plots if the user would
## like to. 
	
	## Intialize a and d
	a = 1
	d = 300

	## Begin figure
	fig,ax = plt.subplots()
	line, = plt.plot(data[:,0],a*func((data[:,0],n(polymer,data[:,0])),d),linewidth=1)

	## Make labels and stuff
	ax.set_xlabel("Wavelength (nm)")
	ax.set_ylabel("Absorbance (au)")
	ax.set_title("Absorbance vs. Wavelength")
	
	## Load color wheel so fit matches data
	color = iter(plt.cm.rainbow(np.linspace(0, 1, num_of_scans)))	
	
	## Plot the data
	for i in range(num_of_scans):
		c = next(color)
		plt.scatter(data[:,i*2],data[:,i*2+1],label=header[0,i*2],s=0.5,color=c)

	## Make room for sliders
	plt.subplots_adjust(left=0.25, bottom=0.25)
	
	## Make horizontal slider for thickness
	axthick = plt.axes([0.25, 0.1, 0.65, 0.03])
	thick_slider = Slider(
	    ax=axthick,
	    label="Thickness (nm)",
	    valmin=0,
	    valmax=1000,
	    valinit=300,
	)

	## Make vertical slider for scale
	axscale = plt.axes([0.1, 0.25, 0.0225, 0.63])
	scale_slider = Slider(
	    ax=axscale,
	    label="Scale (au)",
	    valmin=0,
	    valmax=2,
	    valinit=1,
	    orientation="vertical"
	)
	
	def update(val):
	## Updates the y-values to match the slider

	    line.set_ydata(scale_slider.val * func((data[:,0],n(polymer,data[:,0])),thick_slider.val))
	    fig.canvas.draw_idle()
	
	## Initialize the sliders
	thick_slider.on_changed(update)
	scale_slider.on_changed(update)	
	
	fig = plt.gcf()

	plt.show()	

	## Save the plots to folder
	y_or_n = input("Is this fit satisfactory?(y/n) ")

	## "man" stands for "manual"
	if y_or_n == "y":
		fig.savefig(file_name[:-4] + "_man.pdf", dpi=200)
		return y_or_n

	else:
		return y_or_n


def n(polymer,wavelength):
## This function takes the inputted polymer and wavelength,
## and then returns the wavelength dependent index of refraction
## as an appropriately long vector.

	## Use cauchy formula and data from literature to extract index of refraction
	if polymer == "PVC":
		return np.ones(wavelength.shape[0]) * 1.531
	if polymer == "PS":
		return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35e8) / (wavelength ** 4))
	if polymer == "PMMA":
		return np.ones(wavelength.shape[0]) * 1.482
	#if polymer == "PET":
	#if polymer == "PP":
	#if polymer == "PTFE":

	
def normalize(data,num_of_scans):
## This function uses the relative min and max of the data
## to then adjust to match model better
	
	## Choose how many data points to leave out. 100-300 is good
	cut = 200

	## Ignore the noisy high wavelength stuff
	temp1 = data[cut:,:]

	## Iterate through each scan
	for j in range(num_of_scans):
		
		## Smooth out scan using scipy module
		temp2 = savgol_filter(temp1[:,2*j+1],61,2)

		## Plot the smoothed data
		plt.figure()
		plt.plot(temp1[:,2*j],temp1[:,2*j+1],label="Data")
		plt.plot(temp1[:,2*j],temp2,label="Smoothed")
		plt.title("Raw and Smoothed Data")
		plt.legend()

		## Find index for local min and max
		ind_max = argrelmax(temp2,order=30)[0]
		ind_min = argrelmin(temp2,order=30)[0]

		## Get data points for the max and min locations
		minwav = temp1[:,2*j][ind_min]
		minabs = temp1[:,2*j+1][ind_min]
		maxwav = temp1[:,2*j][ind_max]
		maxabs = temp1[:,2*j+1][ind_max]
	
		## Ignore cases where there is only one min
		if len(ind_min) > 1:

			## Fit to Rayleigh scattering (~1/x^4) or something else
			#coef1 = np.polyfit(minwav**(-4),minabs,1) ## Rayleigh
			#fit1 = coef1[0] * (data[:,2*j]**(-4)) + coef1[1]
			
			#coef1 = np.polyfit(minwav,minabs,1) ## Linear
			#fit1 = coef1[0] * data[:,2*j] + coef1[1]
			
			coef1 = np.polyfit(minwav,np.log(minabs),1) ## Decaying exponential
			fit1 = np.exp(coef1[1])*np.exp(coef1[0]*data[:,2*j])

			#coef1 = np.polyfit(minwav**(-0.5),minabs,1) ## 1/sqrt(2)
			#fit1 = coef1[0] * (data[:,2*j]**(-0.5)) + coef1[1]
			
			## Plot to verify that it is working correctly
			plt.figure()
			plt.plot(temp1[:,2*j],temp1[:,2*j+1],label="Data")
			plt.plot(temp1[:,2*j],temp2,label="Smoothed")
			plt.plot(data[:,2*j],fit1,label="Rayleigh")
			plt.scatter(minwav,minabs,c="b")
			plt.scatter(maxwav,maxabs,c="b")
			plt.legend()

			## Now subtract the fit
			data[:,2*j+1] -= fit1

			## Ignore cases where there is only one max
			if len(ind_max) > 1:
				
				## Update max values for absorbance
				maxabs = temp1[:,2*j+1][ind_max]
	
				## Fit to something that works
				#coef2 = np.polyfit(-maxwav**(-4),maxabs,1) ## Inverse of Rayleigh
				#fit2 = - coef2[0] * (data[:,2*j]**(-4)) + coef2[1]
				
				coef2 = np.polyfit(maxwav,maxabs,1) ## Linear
				fit2 = coef2[0] * data[:,2*j] + coef2[1]

				#coef2 = np.polyfit(maxwav,-np.log(maxabs),1) ## Inverted decaying exponential
				#fit2 = -np.exp(coef2[1])*np.exp(coef2[0]*data[:,2*j])

				#coef2 = 

				scale = np.max(maxabs) / fit2
				
				## Use fit to scale data
				temp3 = np.copy(data[:,2*j+1])
				data[:,2*j+1] *= scale
		
				## Plot to verify it works
				plt.figure()
				plt.plot(data[:,2*j],data[:,2*j+1],label="Corrected Data")
				plt.plot(data[:,2*j],temp3,label="Pre Correction")
				plt.plot(data[:,2*j],fit2)
				plt.scatter(maxwav,maxabs)
				plt.title("Corrected Data")
				plt.legend()
			
			plt.show()
				

		## Otherwise do subtraction using minumum 		
		else:
			temp = data[cut:,:]
			data[:,2*j+1] -= np.amin(temp[:,2*j+1])
	
	return data	

	
def main():

	## Initialize
	polymer = init()

	## Confirm that polymer is supported by code
	if polymer == False:
		return
	
	## Read file
	file_name = input("Please provide file name. ")
	header, data, num_of_scans = read_file(file_name)
	
	## Enter directory to place plots	
	os.chdir("./Plots/Manual")
	
	## Iterate until happy with fits
	y_or_n = "n"
	while y_or_n != "y": 
		
		y_or_n = plot_spectra(header, data, num_of_scans, file_name, polymer)

	return

main()

