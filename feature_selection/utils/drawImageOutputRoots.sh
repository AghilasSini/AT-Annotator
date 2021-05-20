#!/bin/bash 


echo ""
jsonFilename=${1} # full path name
outPng=${2} #full path name

roots2relationgraph $jsonFilename |dot -o$outPng -Tpng

