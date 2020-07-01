[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Medial Axis Detection of a Moving Object

Website: [adityanathan.github.io/projects](https://adityanathan.github.io/projects)

## Prerequisites

- python
- scipy
- opencv
- imutils
- numpy
- skimage

## Steps to Execute Program

- Clone this repository and `cd` to the project directory
- Store a video in the `Videos` directory with a number as the filename. Some sample videos have been provided in the `Videos` directory
- Execute the following command,
    
        python Median_Axis.py -v $NUMBER
    
    where NUMBER = Filename of the video which is to be processed (should be a number)

    Example:
        
        python Median_Axis.py -v 1
- The video will be processed and the output video will be stored in `Videos/Output`

