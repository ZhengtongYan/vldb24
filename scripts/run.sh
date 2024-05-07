#!/bin/bash

# in case the script is not started from within vldb24-repro directory
if [ ! "${PWD}" = "/home/repro/vldb24-reproduction" ]; then
    cd /home/repro/vldb24-reproduction/
fi

cd base

echo "Started Simulated Annealing experiments..."
python3 SimulatedAnnealing.py
echo "Simulated Annealing experiments done."

echo "Started Digital Annealing experiments..."
python3 DigitalAnnealing.py
echo "Digital Annealing experiments done."


#cd /home/repro/vldb24-repro/scripts/plotting
#echo "Plotting DWave results..."
#Rscript dwave_plotting.r
#echo "Plotting done."

cd /home/repro

/bin/bash