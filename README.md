# LiDAR Car Project

**Authors:** Bram & Bela

---

## Overview

This project explores the development of a LiDAR-based perception and navigation system for a small autonomous vehicle.

The goal is to simulate how a car equipped with a LiDAR sensor can:
- perceive its environment,
- detect obstacles,
- and navigate toward a goal.

The project is structured into multiple components, with a strong focus on simulation before moving toward real-world hardware.

---

## Key Idea

We simulate a LiDAR sensor that:
- emits rays in 360 deg
- detects intersections with walls and objects
- produces binary packets similar to a real sensor
- enables navigation using only local perception

---

## Current Capabilities

- Static LiDAR simulation (room scanning)
- Interactive environment builder (furniture + sensor placement)
- Binary LiDAR data generation
- Visualization of point clouds
- Moving vehicle simulation with:
  - obstacle avoidance
  - goal-directed navigation
  - path tracking

---

## Future Work

- Real hardware integration (Raspberry Pi + LiDAR)
- SLAM / mapping
- Global path planning (A*, RRT)
- Improved vehicle dynamics
- Real-time visualization

---

## Getting Started

Run simulations from:

cd simulations
python3 lidar_moving_vehicle.py

Follow the on-screen instructions to:
- draw the room
- place the start position
- place the goal
- run the simulation
