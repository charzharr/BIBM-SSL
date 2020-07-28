
if [ "$USER" == "yzhang46" ]; then

        module load python 
        module load pytorch

        cd /afs/crc.nd.edu/user/y/yzhang46/_BIBM20/src
else
        cd "/Users/charzhar/Desktop/2020 BIBM/project/src"
fi

python -m pytest tests