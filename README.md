# CV Tester
CV tester is a python program to simplify the comparison of different OpenCV feature detectors, keypoint matchers, and usage scenarios.

# About
* Development of this code is ongoing.

* This code currently has only a few protocols to test but can act as a framework to easily add new computer vision techniques such as SURF, BRISK, ORB, or FAST.

* CV Tester is meant for users who would like to test the performance of various feature matchers for their specific use case. CV TEster allows the user to select an image to run a feature detector on and then select what the program should match those keypoints to. Users can attempt to find matches from their webcam stream, a static image stored on their computer, or from a still shot of the webcam feed.

# Using CV Tester
This program needs to be configured directly in the code in order to function properly. All setup parameters are defined at the top of the program.

* The first variable is LIVE. This is a bool which will feature match to a live feed if set to True and will match to a static image if set to False.

* Second is BF. This variable will use the Brute Force feature matcher when set to True and the FLANN matcher if False.

* LOCAL toggles between using a user defined path if set to True and using the programs static image capture from the webcam feed if set to False. If False is chosen for LOCAL, the program will first display a window with the live webcam feed. The user should set the scene how they would like and then hit the "i" key to capture the fram and begin the matching.

* The path1 and path2 variables can be set by the user to point towards images. path1 will always be used as the base image to attempt to match to while path2 is used when LIVE is set to False and LOCAL is set to True.

* Once the program is correctly set up, run the ustility to check the performance of the compiuter vision technique ou selected for your use case.

# Future Updates

* The currently unused Bound() function will be completed, allowing the user to test object detection with a bounding box as an indicator.

* More feature detectors will be added
