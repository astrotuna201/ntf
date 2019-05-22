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
* A single uniform data representation that is used on both the application thread, and transparently used on local and remote accelerators.
* Minimal memory consumption and zero copy capability API variants for “careful” designs
* Convenient expression composition for “casual” prototype designs
* Transparent asynchronous execution model to minimize device stalling and efficiently use collections of devices with continuously variable latencies, both local and remote.
* An execution model that can leverage existing standard driver models such as Cuda and OpenCL.
* Integrated fine grain logging that enables selection of both message level (error, warning, diagnostic) and message category.
* Enable clear closure opportunities for compiler vectorization
* Extensible driver model without requiring a rebuild <TBD>
* Reusable Function repository for rapid model development using “expert” designed functional components.<TBD>

## Execution Model
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
The following is a complete program. It initializes a matrix with a any then takes the sum using the current default device. It doesn't require the user to setup or configure anything.
```swift
let matrix = Matrix<Float>((3, 5), any: 0..<15)
let sum = matrix.sum().scalarValue()
assert(sum == 105.0)
```
This is a simple example of using tensors on the app thread doing normal zip, map, reduce operations.
```swift
// create two tensors and fill with indexes
let a = Matrix<Float>((2, 3), any: 0..<6)
let b = Matrix<Float>((2, 3), any: 6..<12)

let absum = zip(a, b).map { $0 + $1 }

let expected: [Float] = [6, 8, 10, 12, 14, 16]
assert(absum == expected)

let dot = zip(a, b).map(*).reduce(0, +)

assert(dot == 145.0)
```
This selects and sums a 3D sub region on the default device
- initialize a volume with extents (3, 4, 5)
- fill with indexes on device
- on the device create a sub view and take the sum 
- the `scalarValue` function returns the value back to the app thread
```swift
let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
let viewSum = sum(view).scalarValue()
assert(viewSum == 312)
```
If we print with formatting
```swift
print(volume.formatted((2,0)))
```
```
Tensor extents: [3, 4, 5]
  at index: [0, 0, 0]
  -------------------
   0  1  2  3  4 
   5  6  7  8  9 
  10 11 12 13 14 
  15 16 17 18 19 

  at index: [1, 0, 0]
  -------------------
  20 21 22 23 24 
  25 26 27 28 29 
  30 31 32 33 34 
  35 36 37 38 39 

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
Tensor extents: [2, 2, 2]
  at index: [0, 0, 0]
  -------------------
  26 27 
  31 32 

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
The extents of repeated data are not required to match any extent of the Tensor being created. 

These examples repeat row and column vectors.
No matter the extents, `matrix` only uses the shared storage from `rowVector` and repeats it through indexing.
```swift
let rowVector = Matrix<Int32>((1, 5), any: 0..<5)
let rmatrix = Matrix((5, 5), repeating: rowVector)
print(rmatrix.formatted((2,0)))

let colVector = Matrix<Int32>((5, 1), any: 0..<5)
let cmatrix = Matrix((5, 5), repeating: colVector)
print(cmatrix.formatted((2,0)))
```
```
Tensor extents: [5, 5]
at index: [0, 0]
----------------
0  1  2  3  4 
0  1  2  3  4 
0  1  2  3  4 
0  1  2  3  4 
0  1  2  3  4 

Tensor extents: [5, 5]
at index: [0, 0]
----------------
0  0  0  0  0 
1  1  1  1  1 
2  2  2  2  2 
3  3  3  3  3 
4  4  4  4  4 
```
### Matrix Layout
Packages such as Matlab, Octave, and CBLAS have column major memory layout. TensorViews by default are row major, but can work with column major data with zero copy.

This example loads data arranged in column major order.
```swift
//   0, 1,
//   2, 3,
//   4, 5
let matrix = Matrix<Int32>((3, 2),
                           layout: .columnMajor,
                           values: [0, 2, 4, 1, 3, 5])

let expected = [Int32](0..<6)
assert(matrix.array == expected, "values don't match")
```
### Zero Copy Structural Casting of Uniform Dense Scalars
A tensor can store and manipulate structured values. If they are a uniform dense type, they can be structurally recast to other types such as an NHWC tensor used by Cuda.

__<the formatted function needs to be rewritten to perform better type specific output!>__
```swift
let sample = RGBASample<UInt8>(r: 0, g: 1, b: 2, a: 3)
let matrix = Matrix<RGBASample<UInt8>>((2, 3), repeating: Matrix(sample))
let nhwc = NHWCTensor<UInt8>(matrix)
print(nhwc.formatted((2, 0)))
```
```
Tensor extents: [1, 2, 3, 4]
at index: [0, 0, 0, 0]
----------------------
0  1  2  3 
0  1  2  3 
0  1  2  3 

at index: [0, 1, 0, 0]
----------------------
0  1  2  3 
0  1  2  3 
0  1  2  3 
```

### Matrix Zero Copy Transpose
Accessing the MatrixView `t` member variable returns a transposed view of Self with zero copy by manipulating strides.
```swift
let matrix = Matrix<Float>((3, 5), any: 0..<15)
print(matrix.formatted((2,0)))

let tmatrix = matrix.t
print(tmatrix.formatted((2,0)))
```
```
Tensor extents: [3, 5]
at index: [0, 0]
----------------
0  1  2  3  4 
5  6  7  8  9 
10 11 12 13 14 

Tensor extents: [5, 3]
at index: [0, 0]
----------------
0  5  10 
1  6  11 
2  7  12 
3  8  13 
4  9  14 
```
### Result Placement
The biggest peformance problem with all of the major frameworks is copying and constant creation and destruction of tensors. All operator functions and stream functions can take a `result` argument, specifying where the result should be placed, so that temporary variables don't need to be created and destroyed.
```swift
let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

var viewSum = Volume<Int32>((1, 1, 1))
sum(view, result: &viewSum)
assert(viewSum.scalarValue() == 312)
```
### Synchronized Tensor References to Application Buffers 
Tensor constructors are provided to create synchronized references to memory buffers. This is useful to access data from a variety of sources without copying.  The associated _TensorArray_ is initialized with an _UnsafeBufferPointer\<Element\>_ or _UnsafeMutableBufferPointer\<Element\>_ to the data.

Use Examples:
* a record from a memory mapped database or file
* a network data buffer
* a hardware frame buffer
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
let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

let viewSum = using(stream2) {
    sum(view).scalarValue()
}
assert(viewSum == 312)
```

# Logging
Error and diagnostic logging is supported. A log message specifies the _LogLevel_ and diagnostic messages specify _LogCategories_. Each point in the compute platform hierarchy can set a _Log_ object for finer grained reporting. Compute platform objects inherit their parents log. By default the global _Platform_ object defines a log that prints to the console.

Specifying diagnostic categories allows fine grained interest in what is happening, to avoid output overload. Below is an example of how to set logging preferences and related output.
The CPU device uses UMA memory addressing, so a cpuUnitTest service is included that creates discrete memory devices for testing, which is used in the example below.
```swift
Platform.log.level = .diagnostic
Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

let stream1 = Platform.local.createStream(deviceId: 1, serviceName: "cpuUnitTest")
let stream2 = Platform.local.createStream(deviceId: 2, serviceName: "cpuUnitTest")

let volume = using(stream1) {
    Volume<Int32>((3, 4, 5)).filledWithIndex()
}
let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

let viewSum = using(stream2) {
    sum(view).scalarValue()
}
assert(viewSum == 312)
```
Logging shows detailed output of exactly what is happening for the categories specified by the user. If no categories are specified, then diagnostic output is displayed for all categories.
```
status    : default device: [cpu] cpu:0
diagnostic: [CREATE   ] Volume<Int32>(14) Int32[60]
diagnostic: [CREATE   ] Volume<Int32>(15) Int32[60]
diagnostic: [ALLOC    ] Volume<Int32>(15) device array on cpu:1 Int32[60]
diagnostic: [RELEASE  ] Volume<Int32>(14) 
diagnostic: [CREATE   ] Volume<Int32>(17) Int32[1]
diagnostic: [ALLOC    ] Volume<Int32>(17) device array on cpu:2 Int32[1]
diagnostic: [ALLOC    ] Volume<Int32>(15) device array on cpu:2 Int32[60]
diagnostic: [COPY     ] Volume<Int32>(15) cpu:1 --> cpu:2_s0 Int32[60]
diagnostic: [ALLOC    ] Volume<Int32>(17) device array on cpu:0 Int32[1]
diagnostic: [COPY     ] Volume<Int32>(17) cpu:2_s0 --> uma:cpu:0 Int32[1]
diagnostic: [RELEASE  ] Volume<Int32>(17) 
diagnostic: [RELEASE  ] Volume<Int32>(15) 
```
In this example function scheduling and stream synchronization diagnostics are displayed to track exactly what is going on. Object tracking is checked at the end to make sure there are no retain cycles.
```swift
Platform.log.level = .diagnostic
Platform.log.categories = [.dataAlloc, .dataCopy, .scheduling, .streamSync]

do {
    let stream1 = Platform.local.createStream(deviceId: 1, serviceName: "cpuUnitTest")
    let m1 = Matrix<Int32>((2, 5), name: "m1", any: 0..<10)
    let m2 = Matrix<Int32>((2, 5), name: "m2", any: 0..<10)

    // perform on user provided discreet memory stream
    let result = using(stream1) { m1 + m2 }

    // synchronize with host stream and retrieve result values
    let values = try result.array()

    let expected = (0..<10).map { Int32($0 * 2) }
    assert(values == expected)
} catch {
    print(String(describing: error))
}

if ObjectTracker.global.hasUnreleasedObjects {
    print(ObjectTracker.global.getActiveObjectReport())
}
```
Log output
```
status    : default device: [cpu] cpu:0
diagnostic: [CREATE   ] m1(11) Int32[10]
diagnostic: [ALLOCATE ] m1(11) device array on cpu:0 Int32[10]
diagnostic: [CREATE   ] m2(13) Int32[10]
diagnostic: [ALLOCATE ] m2(13) device array on cpu:0 Int32[10]
diagnostic: [CREATE   ] Matrix<Int32>(15) Int32[10]
diagnostic: ~~scheduling: add(lhs:rhs:result:)
diagnostic: [ALLOCATE ] Matrix<Int32>(15) device array on cpu:1 Int32[10]
diagnostic: cpu:1_stream:0 will signal StreamEvent(17) when Matrix<Int32>(15) Int32[10] is complete
diagnostic: [RECORD   ] StreamEvent(17) on cpu:1_stream:0
diagnostic: ~~scheduling: add(lhs:rhs:result:) complete
diagnostic: [WAIT     ] StreamEvent(17) on cpu:0_host:0
diagnostic: [WAIT     ] cpu:0_host:0 will wait for Matrix<Int32>(15) Int32[10]
diagnostic: [ALLOCATE ] Matrix<Int32>(15) device array on cpu:0 Int32[10]
diagnostic: [COPY     ] Matrix<Int32>(15) cpu:1_s0 --> uma:cpu:0 Int32[10]
diagnostic: [RECORD   ] StreamEvent(19) on cpu:0_host:0
diagnostic: [RELEASE  ] Matrix<Int32>(15) 
diagnostic: [RELEASE  ] m2(13) 
diagnostic: [RELEASE  ] m1(11) 
diagnostic: [RECORD   ] StreamEvent(20) on cpu:1_stream:0
diagnostic: [WAIT     ] StreamEvent(20) waiting for cpu:1_stream:0 to complete
diagnostic: [SIGNALED ] StreamEvent(20) on cpu:1_stream:0
```
# Object Tracking
Class objects receive a unique __id__ when registered with the _ObjectTracker_ class. These ids are used in diagnostic messages to know which object is which. 

The _ObjectTracker_ is also used to report unreleased objects to identify retain cycles, and to break the debugger when a specific object instance is created or released. It's much easier and faster to track down problems than using Swift leak detection. During RELEASE builds the tracker does nothing but return id = 0.

The example below creates resources withing a scope. After the scope exits, all resources should be released. The _hasUnreleasedObjects_ property can be used to check for retain cycles in a DEBUG build.
```swift
do {
    Platform.log.level = .diagnostic
    Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

    let stream1 = Platform.local.createStream(deviceId: 1, serviceName: "cpuUnitTest")
    let stream2 = Platform.local.createStream(deviceId: 2, serviceName: "cpuUnitTest")            

    let volume = using(stream1) {
        Volume<Int32>((3, 4, 5)).filledWithIndex()
    }
    let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

    let viewSum = using(stream2) {
        sum(view).scalarValue()
    }
    assert(viewSum == 312)
}

if ObjectTracker.global.hasUnreleasedObjects {
    print(ObjectTracker.global.getActiveObjectReport())
    print("Retain cycle detected")
}
```
