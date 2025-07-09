# Chassis Odometry Update benchmark

Particle filter odometry update with Gaussian noise simulation.

This benchmark tests the performance of updating particle positions using odometry measurements
with added Gaussian noise to simulate sensor uncertainty. Particles are constrained to stay
within specified map boundaries.

The function applies odometry updates to all particles, adds Gaussian noise to simulate
sensor uncertainty, and clips particles to map boundaries. 
