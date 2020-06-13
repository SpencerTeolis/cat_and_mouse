# Overview

A program using a Jetson Nano to track the movements of a cat and direct a laser pointer (mouse) to evade it.

## calibration.py

This is responsible for mapping the image space coordinates to servo coordinates.

## cat-and-mouse.py

This is the main driver file that is responsible for defining the playable space (set by a convex hull around the calibration points for now) and setting the object that is to be tracked. It will then have the laser point avoid that object as well as move towards the object when the object has been immobile.

## mouse-sim-test.py

This is a way to test the "mouse" movement without the camera input by being able to control the "cat" with a cursor. This file can be run without the Jetson hardware if you'd like to test it out yourself.
