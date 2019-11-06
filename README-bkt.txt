Using a Centos 7.5 system running MATE

install visual studio code editor

send ssh keys to github

clone

https://github.com/mq-astrolab/veloce_reduction

which is a fork of Christoph Bergmann's veloce reduction code

install python3

pip3 install virtualenv as root

create a virtualenv for dataredux

virtualenv ~/venvs/dataredux -p /usr/bin/python3


activate it

source ~/venvs/dataredux/bin/activate

prepare some space on the VM, create a file system on the virtual hard drive

mount virtual drive as /data


add veloce data as a share folder on VM

create a vboxsf share on vbox gui and mount it in VM

mount -t vboxsf velocedata /velocedata

veloce_reduction directory - has functions


packages to install:

pip install astropy

repalce veloce_reduction.veloce_reduction with veloce_reduction

or else modules in veloce_reduction subdir wont work

pip install scipy

pip install matplotlib

pip install lmfit

fix impritn of process_scripts

pip install astroquery

pip install barycorrpy

yum install python-matplotlib - for python2


issue with overcommit memory allocation

to fix, echo 1 > /proc/sys/vm/overcommit_memory

also change instances of long to init for python3

comment out bad pixel mask in main_script, doesn't look like the npy that is loaded is used later on


reduced bias frames to 5 fits file

make medbias array int() in line 620 of calibration.py

code is running out of memory ont he poly2d fit step

increased the VM memory to 8 gig

update more int type index issues

use only 5 bias and 5 flats

full set of bias ran out of memory at 4 gb

still running out of memory, create a 32 GB swap in /data

conflicting

no P_id is provided but bg_remove is set to true in process_whites

try just one white flats

fix iterkeys to keys for python3 compatibility

in ordertracing

do
img=img[0,:,:]

for extract_single_stripe function