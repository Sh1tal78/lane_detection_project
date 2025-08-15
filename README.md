# Lane Detection & Departure Warning System

## Description
This project implements a **real-time lane detection and lane departure warning system** using **Python and OpenCV**.  
It processes dashcam video footage to detect road lanes, highlights them, and warns the driver if the vehicle deviates from the lane center.  
This simulates a core feature of **Advanced Driver Assistance Systems (ADAS)** used in modern vehicles.

---

## Features
- Detects **left and right lanes** using Canny edge detection and Hough Transform.
- **Lane departure warning** triggers when the vehicle drifts from the lane.
- Processes **dashcam video input** in real-time.
- Modular, clean, and fully implemented without AI.
- Optional: Can be extended to save output video and display lane curvature.

---

## Tech Stack
- **Python 3.x**
- **OpenCV** (Computer Vision)
- **NumPy** (Numerical computations)

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Sh1tal78/lane_detection_project.git
