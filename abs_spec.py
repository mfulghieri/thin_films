import numpy as np
import matplotlib.pyplot as plt
import os

def init():

	file_name = input("What is the file name? ")

	return file_name

def read_file(file_name):
## This function reads in the .csv file after the file name has 
## been given. From the file, it outputs the name of each scan,
## the respective data, and the number of scans on that file 

	os.chdir("../Data")
        
	header = np.genfromtxt("{0}".format(file_name), delimiter=',', dtype=str, max_rows=2)
    
        ## Get number of scans from the length of the first row, minus the extra comma at the end 
	row1 = header[0,:]
	num_of_scans = int((row1.shape[0] - 1) / 2)

        ## Generate data	
	raw = np.genfromtxt("{0}".format(file_name), delimiter=",", dtype=float, skip_header=2, skip_footer=28*num_of_scans) ## May need to change 27; depends on each file typically

        ## Remove the trailing column of Nan's
	raw = np.delete(raw,num_of_scans * 2,1)
   
	## Average the blanks and the data
	blank_avg = (raw[:,1] + raw[:,3]) / 2
	data_avg = (raw[:,5] + raw[:,7]) / 2	
	diff = data_avg - blank_avg

        ## Subtract out the noise read from the blank scan
	spec = np.zeros((len(diff),2))
	spec[:,0] = raw[:,0]
	spec[:,1] = diff	

	return spec


def plot_spectrum(spec):

	plt.figure()
	plt.plot(spec[:,0],spec[:,1])

	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Absorbance (au)")
	plt.title("Absorbance Spectrum")
		
	plt.tight_layout()
	
	plt.show()

def main():
	
	file_name = init()

	spec = read_file(file_name)

	plot_spectrum(spec)

main()
