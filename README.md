# COMP341-A1
<h2>Project Assignment for Grasp Detection</h2>

Project for the module 'COMP341 - Robot Perception and Manpulation' at the University of Liverpool.
This project demonstrates various neural network implementations for detecting the positions from which a robotic end-effector could pick up an object.
We use the notation [x, y, t, h, w] to define a grasp, where:

<ul>
  <li> x: x-coordinate of grasp centre.</li>
  <li> y: y-coordinate of grasp centre.</li>
  <li> t: angle of rotation. </li>
  <li> h: height of grasp area bounding box.</li>
  <li> w: width of grasp area bounding box.</li>
</ul>

This project uses PyTorch for building neural networks, so please install this before running.

<i>Approaches inspired by the paper 'Real-Time Grasp Detection Using Convolutional Neural Networks' by J. Redmon and A. Angelova, available at https://arxiv.org/pdf/1412.3128.pdf.</i>
