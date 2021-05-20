#!/bin/bash
FILELIST=${1}
INDIR=${2}
BONSAI=/home/aghilas/Workspace/tools/bonsai_v3.2
#export $(BONSAI)

#if [ -f commandFile.txt ]; then
#	rm commandFile.txt
#fi
for infile in `cat $FILELIST`;do
cat >> commandFile.txt << EOF
$BONSAI/bin/bonsai_bky_parse_via_clust.sh -f ldep ${INDIR}/${infile}.txt > ${INDIR}/${infile}.conll
EOF
done


