#!/bin/bash

####################
## Default values ##
####################
export hmm=1
export src=''
export tgt=''
export output=''
export mgiza=''

#####################
## Parse arguments ##
#####################
while getopts ":s:t:o:m:b" opt; do
    case ${opt} in
	s )
	    export src=$OPTARG
	    ;;
	t )
	    export tgt=$OPTARG
	    ;;
	b )
	    export hmm=0
	    ;;
	o )
	    export output=$OPTARG
	    ;;
  m )
    	export mgiza=$OPTARG
    	;;
	\? )
	    >&2 echo "Usage: $0 -s src_file -t tgt_file -o output_file -m mgiza_bin {options}"
	    >&2 echo "options:"
	    >&2 echo "     -b: use IBM Model 1. (Default: use HMM.)"
	    exit 1
	    ;;
    esac
done

if [[ ${src} == '' || ${tgt} == '' || ${output} == '' || ${mgiza} == '' ]];then
  >&2 echo "Usage: $0 -s src_file -t tgt_file -o output_file -m mgiza_bin {options}"
  >&2 echo "options:"
  >&2 echo "     -b: use IBM Model 1. (Default: use HMM.)"
  exit 1
fi

prefix='/tmp/_alignments_'

mkdir -p ${prefix}
text_1=${src}
text_2=${tgt}

name_1=`basename ${text_1}`
name_2=`basename ${text_2}`

echo "Obtaining target and source vocabulary files (.vcb) and sentence pair file (.snt)."

${mgiza}/plain2snt ${text_1} ${text_2} -snt1  ${prefix}/${name_1}_${name_2}.snt -snt2  ${prefix}/${name_2}_${name_1}.snt -vcb1  ${prefix}/${name_1}.vcb -vcb2  ${prefix}/${name_2}.vcb
echo "Done!"
echo "Creating coocurrence file."
${mgiza}/snt2cooc   ${prefix}/${name_1}_${name_2}.cooc  ${prefix}/${name_1}.vcb  ${prefix}/${name_2}.vcb  ${prefix}/${name_1}_${name_2}.snt
${mgiza}/snt2cooc   ${prefix}/${name_2}_${name_1}.cooc  ${prefix}/${name_2}.vcb  ${prefix}/${name_1}.vcb  ${prefix}/${name_2}_${name_1}.snt

echo "Aligning with mgiza."
echo "One way (${name_1} -> ${name_2})"

${mgiza}/mgiza -s ${prefix}/${name_1}.vcb -t  ${prefix}/${name_2}.vcb -c  ${prefix}/${name_1}_${name_2}.snt -coocurrencefile  ${prefix}/${name_1}_${name_2}.cooc -o ${prefix}/${name_1}_${name_2} -hmmdumpfrequency 1

echo "Computing alignments."

if [[ ${hmm} -eq 1 ]]
then
  type='thmm.5'
else
  type='t1.19'
fi

python - ${prefix}/${name_1}.vcb ${prefix}/${name_2}.vcb ${prefix}/*.${type} <<EOF > ${output}
import sys
src = {}
for line in open(sys.argv[1]):
 src[int(line.split()[0])] = line.split()[1]
src[0] = 'NULL'
trg = {}
for line in open(sys.argv[2]):
 trg[int(line.split()[0])] = line.split()[1]
trg[0] = 'NULL'
for line in open(sys.argv[3]):
 print(src[int(line.split()[0])], trg[int(line.split()[1])], line.split()[2])
EOF

rm ${prefix}/${name_1}* ${prefix}/${name_2}*

echo "Done!"