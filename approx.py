import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
import random as r
import os

## Written by Matteo Fulghieri with help from Elizabeth Ryland, Natalia Powers-Riggs, and Trevor Wiechmann

## This code is meant to use an approximation of the thickness using the location of the 
## relative extreme and the index of refraction according to equations from literature. Unlike 
## tester.py, it does not show the data corrections. Further, it is only effective for data that 
## contains more than one relative extremum. Otherwise, MCMC would be recommended.  


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

	## Notice that the normalize step is skipped. The normalizing 
	## procedure actually contains the information necessary to
	## produce the thickenss approximation. I move it to a later 
	## function for simplicity 
	
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


def plot_spectra(header, data, num_of_scans, file_name, t, polymer):
## This function plots the data as a spectrum, the fits, and the 
## output parameters. It also saves the plots if the user would
## like to. 
	
	## Begin figure
	plt.figure()
	ax = plt.gca()
	
	## Load color wheel so fit matches data
	color = iter(plt.cm.rainbow(np.linspace(0, 1, num_of_scans)))	

	## Plot the data, the fit, and parameters
	for i in range(num_of_scans):
		c = next(color)
		plt.scatter(data[:,i*2],data[:,i*2+1],label=header[0,i*2],s=0.5,color=c)
		plt.plot(data[:,i*2],func((data[:,0],n(polymer,data[:,0])),t[i]),linewidth=1,c=c)
		plt.text(0.5,0.9-i*0.05, "{0}: d = {1} nm".format(header[0,i*2],np.around(t[i],2)),transform=ax.transAxes)

	## Plot labels and stuff
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Adjusted Absorbance (a.u.)")
	plt.title("Absorbance vs. Wavelength")
	plt.legend()
	
	plt.tight_layout()
	
	fig = plt.gcf()

	plt.show()	

	## Save the plots to folder
	y_or_n = input("Would you like to save plot?(y/n) ")
	
	## Save the plot if asked to	
	if y_or_n == "y":
		fig.savefig(file_name[:-4] + "_approx.pdf", dpi=200)

	return


def n(polymer,wavelength):
## This function takes the inputted polymer and wavelength,
## and then returns the wavelength dependent index of refraction
## as an appropriately long vector.

	## Use cauchy formula and data from literature to extract index of refraction
	if polymer == "PVC":
		return 1.531
	if polymer == "PS":
		return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35e8) / (wavelength ** 4))
	if polymer == "PET":
		return False
	#if polymer == "PP":
	#if polymer == "PTFE":
	#if polymer == "PMMA":

	
def approximate(data,num_of_scans,polymer):
## This function uses the relative min and max of the data
## to then adjust to match model better
		
	## Initialize object to hold results
	t = np.zeros(num_of_scans)

	## Ignore the noisy high wavelength stuff
	temp1 = data[100:,:]

	## Iterate through each scan
	for j in range(num_of_scans):
		
		## Smooth out scan using scipy module
		temp2 = savgol_filter(temp1[:,2*j+1],61,2)

		## Find index for local min and max
		ind_max = argrelmax(temp2,order=30)[0]
		ind_min = argrelmin(temp2,order=30)[0]

		## Get data points for the max and min locations
		minwav = temp1[:,2*j][ind_min]
		minabs = temp1[:,2*j+1][ind_min]
		maxwav = temp1[:,2*j][ind_max]
		maxabs = temp1[:,2*j+1][ind_max]
		
		## Plot raw and smoothed data, plus extrema
		plt.figure()
		plt.plot(temp1[:,2*j],temp2,label="Smoothed")
		plt.plot(temp1[:,2*j],temp1[:,2*j+1],label="Data")
		plt.scatter(minwav,minabs,c="b")
		plt.scatter(maxwav,maxabs,c="b")
		plt.legend()
		plt.show()

		## Print the values so they can be chosen
		print("minima (in nm): {0}".format(np.around(minwav,2)))
		print("maxima (in nm): {0}".format(np.around(maxwav,2)))

		## Now take in the necessary inputs
		print("Give higher wavelength extrema: ")
		p2 = float(inp())		
		print("Give lower wavelength extrema: ")
		p1 = float(inp())
		print("Give number of cycles (in increments of 0.5): ")
		Ncyc = float(inp())
			
		## Now to calculate the thickness
		v = (n(polymer,p1)/p1 - (n(polymer,p2)/p2))/Ncyc
		d = 1 / (2*v)
		print(np.around(d,2)," nm")	
		t[j] = d
	
	return t

	
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
	os.chdir("./Plots/Approx")
	
	t = approximate(data,num_of_scans,polymer) 
		
	plot_spectra(header, data, num_of_scans, file_name, t, polymer)

	## Print the calculated thickness
	for i in range(num_of_scans):
		print("{0} thickness = ".format(header[0,2*i]), t[i])

	return

main()

