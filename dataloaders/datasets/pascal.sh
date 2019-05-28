#!/bin/bash
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ./data/ ..." 
    cd ./data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Now downloading pascal.tar ..."

# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# The official link is not working for some reason, so temporarily use dropbox instead.

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo "Unzip pascal.tar ..."

tar xvf VOCtrainval_11-May-2012.tar

echo "Removing unnecessary files ..."

rm -f VOCtrainval_11-May-2012.tar
# rm -f Figaro1k/GT/Training/*'(1).pbm'
# rm -f Figaro1k/.DS_Store
# rm -rf __MACOSX

echo "Finished!"
