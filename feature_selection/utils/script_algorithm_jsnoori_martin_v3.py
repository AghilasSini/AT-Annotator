#!/usr/bin/jython

# Version Aghilas Sini de juillet 2016, plus quelques modifs...:
# v2:
#    - deplacement du coeff energie apres les valeurs de pitch, pour avoir le F0 en premier coefficient, juste apres le temps.
#    - temps en secondes
# v3.0:
#    - garde pitch calcule en sortie, et affichage ensuite les 3 candidats F0.
# v3.1:
#    - type locuteur par defaut = "-1" (any speaker) au lieu de "1" (male speaker) dans version precedente.
#    - frame shift par defaut mis a 10 ms.
# v3.2:
#   - options pour ecrire en sortie les candidats f0 (-writeF0Candidates) et les parametres acoustiques (-writeAcouFeatures).
#     par defaut, le fichier de sortie ne comprend que le temps de la trame et le pitch correspondant.



import os
import sys
from java.io import FileOutputStream
import  logging 
import  optparse
import getopt
import string


__all__ = []
__version__ = '3.2'
__date__ = '15-03-2015'
__updated__ ='23-08-2016'
__author__='asini,djouvet'


## related path
if os.path.dirname(sys.argv[0])!= "":
    directory_name=os.path.dirname(sys.argv[0])+"/"
else :
    directory_name="";

directory_jsnoori = directory_name+"Aghilas.July2016/"


#load class (binary path)
os.sys.path.append(directory_name+"bin")
os.sys.path.append(directory_jsnoori+"bin")


#build path directory
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

    
# Run the above function and store its results in a variable.   
jar_files_paths = get_filepaths(directory_name+"lib")
for jarfilename in get_filepaths(directory_jsnoori+"lib"):
    jar_files_paths.append(jarfilename)

    
# load all jar file
for jarfilename in jar_files_paths:
    os.sys.path.append(jarfilename)

    
# import Library
import org.netlib.lapack
from fr.loria.parole.jsnoori.model.speech.pitch import Pitch
from fr.loria.parole.jsnoori.model.speech.pitch import AsyncPitch
from fr.loria.parole.jsnoori.model.speech import Spectrogram
from java.util import Vector
from fr.loria.parole.jsnoori.util.file.segmentation import TextGridSegmentationFileUtils
from fr.loria.parole.jsnoori.model import ResourcePath
from fr.loria.parole.jsnoori.model import JSnooriProperties
from fr.loria.parole.jsnoori.model import Constants
from fr.loria.parole.jsnoori.model.audio import AudioSignal
from fr.loria.parole.jsnoori.model import Constants
from fr.loria.parole.jsnoori.util import Energy
from fr.loria.parole.jsnoori.util import TimeConversion
from fr.loria.parole.jsnoori.model.speech import FeedbackPreProcessing
from fr.loria.parole.jsnoori.model.speech.feature import MFCC
from fr.loria.parole.jsnoori.util import TimeConversion


## Options 
parser=optparse.OptionParser(usage="jython or java lib/jython.jar  %prog [options] arg", version="%prog 3.2")
parser.add_option("-i", "--input",      dest="input",        type="string",      default=None,   help="take input file (only audio wave file .wav), no default input filename")
parser.add_option("-o", "--output",     dest="output",       type="string",      default=None,   help="write output to file (raw text file), no default output filename ")
parser.add_option("--writeF0Candidates",dest="writef0cand",  action="store_true",default=False,  help="add f0 candidates in output file")
parser.add_option("--writeAcouFeatures",dest="writeacoufeat",action="store_true",default=False,  help="add acoustic features in output file")
parser.add_option("--windowSize",       dest="window",       type="int",         default=32,     help="window size (ms), 32 ms is the default value")
parser.add_option("--windowShift",      dest="shift",        type="int",         default=10,     help="window shift (ms), 10 ms is the default")
parser.add_option("--speakerGender",    dest="sexe",         type="int",         default=-1,     help="choose sexe of speakers (0:female, 1:male, -1:any)")
parser.add_option("--dbmin",            dest='dbmin',        type="float",       default=60.0,   help="lowest energy (db)",)
parser.add_option("--fftOrder",         dest='fftOrder',     type="float",       default=1024,   help="FFT order (samples)")
parser.add_option("--cc",               dest='cc',           type="int",         default=40,     help="number of cepstral coefficients")
parser.add_option("--lpc",              dest='lpc',          type="int",         default=20,     help="number of lpc coefficients ")
parser.add_option("--mfcc",             dest='mfcc',         type="int",         default=13,     help="number of mel cepstral coefficients")
parser.add_option("--start",            dest='start',        type="float",       default=0,      help="start time (ms)")


def pretty_printing(cepstral_coef_v):
    cc=[]
    for i in range(0,len(cepstral_coef_v)):
        list_coef_frame=["%.3f"%cepstral_coef_v[i][j] for j in range(0,len(cepstral_coef_v[i]))]  
        cc_frame='\t'.join(map(str,list_coef_frame))   
        # print i,list_coef_frame
        cc.append(cc_frame)
    return cc


#print pitchs result
def __print__(pitchs, candidatesList, energy_v, cepstral_coef_v, ccorder, mel_cepstral_v, melceporder, f0median, fmin, fmax, filename, inputfile, signal, sexe, shift):
    ## cepstral Coeficients
    cc=pretty_printing(cepstral_coef_v)
    #  mel cepstral coeficients
    mfcc=pretty_printing(mel_cepstral_v)
    #open file
    f=open(filename,"w")
    #
    f.write("# Script: " + sys.argv[0] + "  (version " + __version__ + ")\n")
    f.write("# Window:  size = " + str(options.window) + " ms;  shift = " + str(options.shift) + " ms\n")
    f.write("# Sexe: " + str(sexe) + "  (0<=>female, 1<=>male, -1<=>any)\n")
    f.write("# Lowest energy: " + str(options.dbmin) + " dB;  FFT order: " + str(options.fftOrder) + "\n")
    f.write("# Output start time: " + str(options.start) + " ms\n")
    if options.writeacoufeat:
        f.write("# Acoustic features order:  " + str(options.cc) + " cepstral coefs;  " + str(options.lpc) + " lpc coefs;  " + str(options.mfcc) + " mfcc coefs\n")
    f.write("#\n")
    #
    f.write("# Signal file name: "+inputfile+"\n")
    signalinfo=str(signal).split('\n')
    signalinfo=["%s"%c for c in signalinfo]
    signalinfo=';  '.join(map(str,signalinfo))  
    f.write("# " + signalinfo + "\n")
    f.write("# f0 computation info:  min= " + str(fmin) + "Hz;  max= " + str(fmax) + "Hz;  median= " + str(f0median) + "Hz\n")
    f.write("#\n")
    # linear cepstral coeficient(columns names)
    header_cc=["cc%d"%c for c in range(0,ccorder)]
    header_cc='\t'.join(map(str,header_cc))  
    # mel cepstral coeficient (columns names)
    header_mfcc=["mfcc%d"%m for m in range(1,melceporder)]
    header_mfcc='\t'.join(map(str,header_mfcc)) 
    # 
    line = "#time_s\tf0_Hz"
    if options.writef0cand:
        line += "\tf00_hz\tf00_corr\tf01_hz\tf01_corr\tf02_hz\tf02_corr"
    if options.writeacoufeat:
        line += "\tenergy\t%s\t%s"%(header_cc,header_mfcc)
    f.write(line+"\n")
    # 
    t=options.start
    for k in range(0,len(pitchs)):   
        f0_Hz    = pitchs[k][1]
        line     = '%.3f\t%.1f'%(t/1000.,f0_Hz)  
        if options.writef0cand:
            f00_corr = candidatesList.get(k).fst[0]
            f00_hz   = candidatesList.get(k).snd[0]
            f01_corr = candidatesList.get(k).fst[1]
            f01_hz   = candidatesList.get(k).snd[1]
            f02_corr = candidatesList.get(k).fst[2]
            f02_hz   = candidatesList.get(k).snd[2]
            line    +='\t%.1f\t%.3f\t%.1f\t%.3f\t%.1f\t%.3f'%(f00_hz,f00_corr,f01_hz,f01_corr,f02_hz,f02_corr)  
        if options.writeacoufeat:
            line    +='\t%.3f\t%s\t%s'%(energy_v[k],cc[k],mfcc[k])  
        f.write(line+"\n")
        t+=shift  
    f.close()

    
# check options
try:
    (options, args)=parser.parse_args()
    if options.input==None:
        parser.error('this option cannot be empty')
except Exception, e:
    raise e
    return options,args

try:
    inputfile=options.input
except Exception, e:
    inputfile=None
    

# Processing
if  inputfile!=None:   
    # load wave signal
    signal=AudioSignal(inputfile)
    # resize the signal
    firstSample=(int)(TimeConversion.enEchtf(options.start,signal))
    if firstSample!=0:
        new_size=signal.getSampleCount()-firstSample
        samples=signal.getSamples(firstSample,new_size)
        signal.setSignal(signal.getSampleRate(), samples)
    # pitch time shift
    timeShift=options.shift
    # pitch Window
    window=options.window
    # Pitch's Object
    pitch=Pitch(options.window,timeShift)
    # male: 1; female: 0; unknow: -1.
    sexe=options.sexe
    # compute pitchs 
    pitchs=pitch.computePitch(signal,sexe)
    # compute median F0
    f0median=pitch.pitchMedian()
    # get  f0 minmale
    fmin=pitch.getF0Min()
    # get f0 maximale
    fmax=pitch.getF0Max()
    # F0 candidates
    candidatesList=pitch.getPitchs()
    # pitch size
    pitch_count=len(candidatesList)
    # Conversion frome time to samples
    sampleSift=int(TimeConversion.enEchtf(timeShift, signal))
    # new window(samples)
    window_samples=signal.getSampleCount()-(sampleSift*pitch_count)
    # conversion
    window_ms=int(TimeConversion.enMillif(window_samples,signal))
    # compute energy
    energy=Energy(signal,fmin,fmax,options.dbmin,window_ms,options.fftOrder,timeShift)
    # cepstral coefficients
    cepstral_coef=FeedbackPreProcessing(signal,options.lpc, options.fftOrder, window_ms,timeShift, options.cc, True, False);
    # samplingRate
    Fs=int(signal.getSampleRate())
    # mel frequency cepstral coeficient.
    mfcc =MFCC(window_samples, Fs, options.mfcc,options.fftOrder)
    #
    energy_v=[]
    cepstral_coef_v=[]
    mel_cepstral_coef_v=[]

    for x in range(0,pitch_count):
        # frame 
        frame=energy.framing(x, sampleSift,window_samples,0)
        # absolute energy for given frame
        energy_v.append(energy.computeEnergyForSingleFrame(frame,window_samples))
        # cepstral coefficients for given frame
        cepstral=list(cepstral_coef.cepstralCoefForSingleFrame(x,signal))
        # 
        cepstral_coef_v.append(cepstral)
        # 
        mel_cepstral_coef_v.append(mfcc.doMFCC(frame)[1:])

    # print output file
    if(options.output!=None):    
         __print__(pitchs, candidatesList, energy_v, cepstral_coef_v, options.cc, mel_cepstral_coef_v, options.mfcc, f0median, fmin, fmax, options.output, options.input, signal, sexe, timeShift)       
         
else:
    print "error"


#
# End of file.
#
