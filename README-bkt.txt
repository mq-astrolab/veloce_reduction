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

