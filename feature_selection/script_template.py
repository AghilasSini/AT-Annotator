#!/usr/bin/python3
import sys
import os
import argparse
import numpy as np 



def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('inputFile', type=str, nargs=1, help='name of the input file')
	parser.add_argument('outputFile',type=str, nargs=1, help='name of the output file')
	# option
	parser.add_argument('--opt_name',type=str,dest="destination",default="default name",help='')
	args = parser.parse_args()
	inputFilename=args.inputFile[0]
	


if __name__== "__main__":
	main()
