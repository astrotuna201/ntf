# NetlibTF

## Overview
This project extracts some key Netlib IP that I have reworked and enhanced specifically to meet the design goals of the Google Swift 4 TensorFlow project as I understand them.

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

## Simple Examples of the Experience
The following is a complete program. It initializes a matrix with a sequence then takes the sum. It uses shortcut syntax to specify the matrix extents (3, 5)
```swift
let matrix = Matrix<Float>(3, 5, sequence: 0..<15)
let sum = matrix.sum().scalarValue()
assert(sum == 105.0)
```

        //--------------------------------
        // Select and sum a 3D sub region
        // - initialize a volume using explicit extents
        // - fill with indexes on the default device
        // - take the sum of the sub view on the device
        // - return the scalar value back to the app thread
        do {
            let volume = Volume<Int32>(extents: [3, 4, 5]).filledWithIndex()
            print(volume.formatted(scalarFormat: (2,0)))
            
            let sample = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            print(sample.formatted(scalarFormat: (2,0)))
            
            let sampleSum = sum(sample).scalarValue()
            XCTAssert(sampleSum == 312)
        }
        
        //--------------------------------
        // repeat a value
        // No matter the extents, `volume` only uses the shared storage
        // from `value` and repeats it through indexing
        do {
            let volume = Volume<Int32>(extents: [2, 3, 10],
                                       repeating: Volume(42))
            print(volume.formatted(scalarFormat: (2,0)))
        }
        
        //--------------------------------
        // repeat a vector
        // No matter the extents, `matrix` only uses the shared storage
        // from `rowVector` and repeats it through indexing
        do {
            let rowVector = Matrix<Int32>(1, 10, sequence: 0..<10)
            let rmatrix = Matrix(extents: [10, 10], repeating: rowVector)
            print(rmatrix.formatted(scalarFormat: (2,0)))

            let colVector = Matrix<Int32>(10, 1, sequence: 0..<10)
            let cmatrix = Matrix(extents: [10, 10], repeating: colVector)
            print(cmatrix.formatted(scalarFormat: (2,0)))
        }
