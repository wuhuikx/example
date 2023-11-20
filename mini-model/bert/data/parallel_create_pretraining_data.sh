# export VOCAB_FILE=/data/wiki/bert-base-uncased-vocab.txt
cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
if [ ${cpus} -gt 64 ]; then
cpus=64
fi
echo "Using ${cpus} CPU cores..."
datadir=$1
find -L ${datadir} -name "pretrain-part*" | xargs --max-args=1 --max-procs=${cpus} bash create_pretraining_data.sh
