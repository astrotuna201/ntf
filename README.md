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

The idea of a stream of commands going to a local or remote device across the network seems easy for users to understand and it fits well with encapsulating frameworks like Cuda or OpenCL.

***
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

For example: if a batch of 64 image items is loaded, they can all be decoded and written to a tensor using independent threads making full use of all cores on the host CPU. Using sub-views in parallel on the GPU works the same way. Synchronization between writable sub views is managed by the caller.

__<I need to redo this diagram to change the names!>__

![TensorArray Diagram](https://github.com/ewconnell/Netlib/blob/master/documents/DataSubViewDiagram.png)

Tensor views manage both shared and copy-on-write semantics. Multiple concurrent views can be created to support multi-threaded access. The caller manages synchronization between writable views.


## Simple Use Examples
The following is a complete program. It initializes a matrix with a sequence then takes the sum using the current default device. It doesn't require the user to setup or configure anything.
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
print(volume.formatted((2,0)))
```
```
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
print(subView.formatted((2,0)))
```
```
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
print(volume.formatted((2,0)))
```        
Repeating any pattern whether it matches any dimensions is allowed. These repeat a row and column vectors.
No matter the extents, `matrix` only uses the shared storage from `rowVector` and repeats it through indexing.
```swift
let rowVector = Matrix<Int32>((1, 10), sequence: 0..<10)
let rmatrix = Matrix((10, 10), repeating: rowVector)
print(rmatrix.formatted((2,0)))

let colVector = Matrix<Int32>((10, 1), sequence: 0..<10)
let cmatrix = Matrix((10, 10), repeating: colVector)
print(cmatrix.formatted((2,0)))
```
```
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

let expected: [Int32] = [
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0,  1,  2, -1, -1, -1,
    -1, -1,  3,  4,  5, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
]

assert(matrix.array == expected, "values do not match")
```
### Matrix Layout
Packages such as Matlab, Octave, and CBLAS have column major memory layout. TensorViews by default are row major, but can work with column major data with zero copy.

This example loads data arranged in column major order.
```swift
//   0, 1,
//   2, 3,
//   4, 5
let cmMatrix = Matrix<Int32>((3, 2),
                             layout: .columnMajor,
                             scalars: [0, 2, 4, 1, 3, 5])

let expected = [Int32](0..<6)
assert(cmMatrix.array == expected, "values don't match")
```
### Matrix Zero Copy Transpose
Accessing the MatrixView _t_ member variable returns a transposed view of Self with zero copy by manipulating strides.
```swift
let matrix = Matrix<Float>((3, 5), sequence: 0..<15)
print(matrix.formatted((2,0)))

let tmatrix = matrix.t
print(tmatrix.formatted((2,0)))
```
```
TensorView extents: [3, 5] paddedExtents: [3, 5]
at index: [0, 0]
----------------
0  1  2  3  4 
5  6  7  8  9 
10 11 12 13 14 

TensorView extents: [5, 3] paddedExtents: [5, 3]
at index: [0, 0]
----------------
0  5 10 
1  6 11 
2  7 12 
3  8 13 
4  9 14 
```
### Result Placement
The biggest peformance problem with all of the major frameworks is copying and constant creation and destruction of tensors. All operator functions and stream functions will take a `result` argument, specifying where the result should be placed, so that temporary variables are not created and destroyed unnecessarily.
```swift
let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

var subViewSum = Volume<Int32>((1, 1, 1))
sum(subView, result: &subViewSum)
assert(subViewSum.scalarValue() == 312)
```

***
# Device Abstraction
NetlibTF defines a set of class protocols for platform abstraction. They encapsulate functionality to allow run time selection of a compute service (cpu, cuda, metal, etc.) and hardware device, to enable application portability without recoding. 

ComputePlatform
      services[]
       ComputeService (cpu, cuda, amd, tpu, ...)
         devices[]
           ComputeDevice (gpu:0, gpu:1, ...)
             DeviceArray
             DeviceStream
           StreamEvent

Concrete implementations are provided. The _Platform_ class is the root for local resource selection and allocation. A _RemotePlatform_ class and marshalling objects are planned.

### ComputePlatform protocol
The _ComputePlatform_ is used to select a compute service (cpu, cuda, metal, etc.), hardware device, and to specify a default device. The _ComputePlatform_ is also used to detect and load compute service plugins that are implemented in separate bundles. This permits dynamic download and use of compute drivers without recompiling the application.

__<The Linux Foundation library didn't have loadable bundles when I last checked. Recheck!>__

### ComputePlatform
A compute platform represents the root for managing all services, devices, and streams on a platform. There is one local instance per process, and possibly many remote instances.
### ComputeService
A compute service implements the _ComputeService_ protocol and is used to enumerate available devices.
### ComputeDevice
A compute device implements the _ComputeDevice_ protocol and is used to query device attributes, and create resources such as device arrays and streams.
### DeviceStream
A device stream is an abstraction for an asynchronous command queue that implements the _DeviceStream_ protocol. It is used to schedule and synchronize computations. The protocol function implementations are service API specific and optimized.
### DeviceArray
A device array implements the _DeviceArray_ protocol and is an abstraction for a contiguous array of bytes on a device. 
### StreamEvent
A stream event implements the _StreamEvent_ protocol and is used to synchronize device streams.

# Using Multiple Streams and Devices
By default a stream is created for the user on the current thread in the global scope. Operations are implicitly performed on this stream. However more sophisticated users will want to create multiple streams on multiple devices on multiple platforms. The syntax for using streams is straitforward.
```swift
let stream1 = Platform.local.createStream(deviceId: 1)
let stream2 = Platform.local.createStream(deviceId: 2)

let volume = using(stream1) {
    Volume<Int32>((3, 4, 5)).filledWithIndex()
}
let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

let subViewSum = using(stream2) {
    sum(subView).scalarValue()
}
assert(subViewSum == 312)
```

# Logging
Error and diagnostic logging is supported. A log message specifies the _LogLevel_ and diagnostic messages specify _LogCategories_. Each point in the compute platform hierarchy can set a _Log_ object for finer grained reporting. Compute platform objects inherit their parents log. By default the global _Platform_ object defines a log that prints to the console.

Specifying diagnostic categories allows fine grained interest in what is happening, to avoid output overload. Below is an example of how to set logging preferences and related output.
The CPU device uses UMA memory addressing, so a cpuUnitTest service is included that creates discrete memory devices for testing, which is used in the example below.
```swift
Platform.log.level = .diagnostic
Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

let stream1 = Platform.local.createStream(serviceName: "cpuUnitTest", deviceId: 1)
let stream2 = Platform.local.createStream(serviceName: "cpuUnitTest", deviceId: 2)

let volume = using(stream1) {
    Volume<Int32>((3, 4, 5)).filledWithIndex()
}
let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

let subViewSum = using(stream2) {
    sum(subView).scalarValue()
}
assert(subViewSum == 312)
```
Logging shows detailed output of exactly what is happening for the categories specified by the user. If no categories are specified, then diagnostic output is displayed for all categories.
```
status    : default device: [cpu] cpu:0
diagnostic: [CREATE ] Volume<Int32>(14) elements[60]
diagnostic: [CREATE ] Volume<Int32>(15) elements[60]
diagnostic: [ALLOC  ] Volume<Int32>(15) device array on cpu:1 elements[60]
diagnostic: [RELEASE] Volume<Int32>(14) elements[60]
diagnostic: [CREATE ] Volume<Int32>(17) elements[1]
diagnostic: [ALLOC  ] Volume<Int32>(17) device array on cpu:2 elements[1]
diagnostic: [ALLOC  ] Volume<Int32>(15) device array on cpu:2 elements[60]
diagnostic: [COPY   ] Volume<Int32>(15) cpu:1 --> cpu:2_s0 elements[60]
diagnostic: [ALLOC  ] Volume<Int32>(17) host array elements[1]
diagnostic: [COPY   ] Volume<Int32>(17) cpu:2_s0 --> host elements[1]
diagnostic: [RELEASE] Volume<Int32>(17) elements[1]
diagnostic: [RELEASE] Volume<Int32>(15) elements[60]
```
# Object Tracking
Class objects receive a unique __id__ when registered with the _ObjectTracker_ class. These ids are used in diagnostic messages to know which object is which. 

The _ObjectTracker_ is also used to report unreleased objects to identify retain cycles, and to break the debugger when a specific object instance is created or released. It's much easier and faster to track down problems than using Swift leak detection. During RELEASE builds the tracker does nothing but return an id from a simple counter.
