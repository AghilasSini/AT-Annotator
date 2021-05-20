
#
INDIR=${1}
FILELIST=${2}

for infile in `cat $FILELIST`;do
cat >> commandFile.txt << EOF
	python3.5 linguistic_features_extraction.py ${INDIR}${infile}
EOF
done