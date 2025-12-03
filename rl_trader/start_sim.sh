#!/bin/bash

cd /home/shiftpub/shift-main
./startup.sh -r
InitializationProgram -b 1 -e 100 -t CS1 -p 100 -s 10000 < /dev/null > /dev/null 2>&1
InitializationProgram -b 101 -e 200 -t CS2 -p 100 -s 10000 < /dev/null > /dev/null 2>&1

cd /home/shiftpub/shift-research/agents/ZITrader
./batch2.sh