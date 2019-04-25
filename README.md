# NetlibTF

## Overview
NetlibTF is key Netlib IP that has been reworked and enhanced specifically to meet the design goals of the Google Swift 4 TensorFlow project (as I understand them).

The design currently addresses:
* Tensor representation
* Device abstraction
* Asynchronous execution
* Logging

The code is in early stages and needs signficant testing along with performance and usuability refinement.

The code will compile inside the S4TF environment, but currently there are no dependencies on TensorFlow. Some names are not the desired name, for example TensorShape is called DataShape and Tensor is called NDTensor to avoid naming conflicts.

## Design Goals
* Optimal defaults for all configuration parameters so the user can have a good experience with no upfront training or special effort
* Simplified local and remote compute device management for more sophisticated applications
* A single uniform data representation that is used on both the application thread in “Swift space”, and transparently used on local and remote accelerators.
* Minimal memory consumption and zero copy capability API variants for “careful” designs
* Convenient expression composition for “casual” prototype designs
* Transparent asynchronous execution model to minimize device stalling and efficiently use collections of devices with continuously variable latencies, both local and remote.
* An execution model that can leverage existing standard driver models such as Cuda and OpenCL.
* Integrated fine grain logging that enables selection of both message level (error, warning, diagnostic) and message category.
* Enable clear closure opportunities for compiler vectorization
* Extensible driver model without requiring a rebuild <TBD>
* Reusable Function repository for rapid model development using “expert” designed functional components.<TBD>

## Proposed Execution Model
The design goal is to have an asynchronous execution model that is transparent to the user and can leverage existing driver infrastructure such as Cuda, OpenCL, and other proprietary models such as Google TPUs.
I propose adopting an asynchronous stream based driver model to meet this goal for both local and remote devices.

# Tensor Representation
A tensor is a dynamically sized n-dimensional data array. The Tensor (NDTensor) type can be manipulated much the same as the TensorFlow tensor type. This is flexible, but can make user code harder to understand. Therefore shaped types are provided for clarity and to offer type specific initializers and helper functions. The types currently defined are:
* ScalarValue
* Vector
* Matrix
* Volume
* Tensor (NDTensor)
* NHWC
* NCHW

All of these types are just constrained variations of a _Tensor_ which conforms to the _TensorView_ protocol. All underlying operators and driver functions require conformance to _TensorView_ and not to shaped types. Operator arguments are handled as n-dimensional data sets.

## Tensor Structure
### TensorArray
A _TensorArray_ is an abstract representation of a contiguous fixed size linear byte array. No data space is actually allocated until the first access is made. The point of access determines where the data is allocated. So if data is first accessed on a device, it will only exist there unless referenced somewhere else, making on device temporary variables efficient.

__<I need to redo this diagram to change the names!>__

![TensorArray Diagram](https://github.com/ewconnell/Netlib/blob/master/documents/DataArrayDiagram.png)

### TensorView
A _TensorView_ is a struct that presents a shaped view of an associated _TensorArray_ object, along with a variety of access functions. Creation of a _TensorView_ will automatically create a _TensorArray_ object if one is not specified.

### Sub Views and App Space Multi-threading
_TensorView_ has methods to create sub views of a _TensorArray_. An important use of this is to divide a _TensorArray_ into multiple sub-regions and operate on them in parallel. 

For example: if a batch of 64 image items is loaded, they can all be decoded and written to the tensor on independent threads making full use of all cores on the host CPU. Using sub-views in parallel on the GPU works the same way. Synchronization between writable sub views is managed by the caller.

__<I need to redo this diagram to change the names!>__

![TensorArray Diagram](https://github.com/ewconnell/Netlib/blob/master/documents/DataSubViewDiagram.png)

Tensor views manage both shared and copy-on-write semantics. Multiple concurrent views can be created to support multi-threaded access. The caller manages synchronization between writable views.


## Simple Use Examples
The following is a complete program. It initializes a matrix with a sequence then takes the sum. It doesn't require the user to setup or configure anything.
```swift
let matrix = Matrix<Float>((3, 5), sequence: 0..<15)
let sum = matrix.sum().scalarValue()
assert(sum == 105.0)
```

This selects and sums a 3D sub region
- initialize a volume using explicit extents
- fill with indexes on the default device
- on the device create a sub view and take the sum 
- return the scalar value back to the app thread
```swift
let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
let subViewSum = sum(subView).scalarValue()
assert(subViewSum == 312)
```
If we print with formatting
```swift
print(volume.formatted(scalarFormat: (2,0)))
```
```sh
TensorView extents: [3, 4, 5] paddedExtents: [3, 4, 5]
at index: [0, 0, 0]
===================
  at index: [0, 0, 0]
  -------------------
   0  1  2  3  4 
   5  6  7  8  9 
  10 11 12 13 14 
  15 16 17 18 19 

at index: [1, 0, 0]
===================
  at index: [1, 0, 0]
  -------------------
  20 21 22 23 24 
  25 26 27 28 29 
  30 31 32 33 34 
  35 36 37 38 39 

at index: [2, 0, 0]
===================
  at index: [2, 0, 0]
  -------------------
  40 41 42 43 44 
  45 46 47 48 49 
  50 51 52 53 54 
  55 56 57 58 59 
```
```swift
print(subView.formatted(scalarFormat: (2,0)))
```
```sh
TensorView extents: [2, 2, 2] paddedExtents: [2, 2, 2]
at index: [0, 0, 0]
===================
  at index: [0, 0, 0]
  -------------------
  26 27 
  31 32 

at index: [1, 0, 0]
===================
  at index: [1, 0, 0]
  -------------------
  46 47 
  51 52 
```
All tensor views are able to repeat data through indexing. No matter the extents, `volume` only uses storage
for a single value.
```swift
let volume = Volume<Int32>((2, 3, 10), repeating: Volume(42))
print(volume.formatted(scalarFormat: (2,0)))
```        
Repeating any pattern whether it matches any dimensions is allowed. These repeat a row and column vectors.
No matter the extents, `matrix` only uses the shared storage from `rowVector` and repeats it through indexing.
```swift
let rowVector = Matrix<Int32>((1, 10), sequence: 0..<10)
let rmatrix = Matrix((10, 10), repeating: rowVector)
print(rmatrix.formatted(scalarFormat: (2,0)))

let colVector = Matrix<Int32>((10, 1), sequence: 0..<10)
let cmatrix = Matrix((10, 10), repeating: colVector)
print(cmatrix.formatted(scalarFormat: (2,0)))
```
```sh
TensorView extents: [10, 10] paddedExtents: [10, 10]
at index: [0, 0]
----------------
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 
 0  1  2  3  4  5  6  7  8  9 

TensorView extents: [10, 10] paddedExtents: [10, 10]
at index: [0, 0]
----------------
 0  0  0  0  0  0  0  0  0  0 
 1  1  1  1  1  1  1  1  1  1 
 2  2  2  2  2  2  2  2  2  2 
 3  3  3  3  3  3  3  3  3  3 
 4  4  4  4  4  4  4  4  4  4 
 5  5  5  5  5  5  5  5  5  5 
 6  6  6  6  6  6  6  6  6  6 
 7  7  7  7  7  7  7  7  7  7 
 8  8  8  8  8  8  8  8  8  8 
 9  9  9  9  9  9  9  9  9  9 
```
Virtual padding can be specified for a view. Here one padding row is added before and after, and columns are padded 2 before and 3 after. A padding value of -1 is used here to make boundaries obvious. The default padding value is 0. Padding can also be added to sub views to aid windowed operations such as convolutions.
```swift
let padding = [
    Padding(1),                   // row pad
    Padding(before: 2, after: 3)  // col pad
]
let matrix = Matrix<Int32>((2, 3),
                           padding: padding,
                           padValue: -1,
                           sequence: 0..<6)

let expectedValues: [Int32] = [
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0,  1,  2, -1, -1, -1,
    -1, -1,  3,  4,  5, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
]
let values = [Int32](matrix.values())
assert(values == expectedValues, "indices do not match")
```

