# NetlibTF

## Overview
This project extracts some key Netlib IP and reworks it specifically to meet the design goals of the Google Swift 4 TensorFlow project as I understand them.

The code is in early stages and needs signficant testing along with performance and usuability refinement.

## Core Design Goals
* Optimal defaults for all configuration parameters so the user can have a good experience with no upfront training or special effort
* Simplified local and remote compute device management for more sophisticated applications
* A single uniform data representation that is used on both the application thread in “Swift space”, and transparently used on local and remote accelerators.
* Minimal memory consumption and zero copy capability API variants for “careful” designs
* Convenient expression composition for “casual” prototype designs
* Transparent asynchronous execution model to minimize device stalling and efficiently use collections of devices with continuously variable latencies, both local and remote.
* An execution model that can leverage existing standard driver models such as Cuda and OpenCL.
* Integrated fine grain logging that enables selection of both message level (error, warning, diagnostic) and message category.
* Enable clear closure opportunities for compiler vectorization
* Extensible driver model without requiring a rebuild
* Reusable Function repository for rapid model development using “expert” designed functional components.

## Proposed Execution Model
The design goal is to have an asynchronous execution model that is transparent to the user and can leverage existing driver infrastructure such as Cuda, OpenCL, and other proprietary models such as Google TPUs.
I propose adopting an asynchronous stream based driver model to meet this goal for both local and remote devices.
