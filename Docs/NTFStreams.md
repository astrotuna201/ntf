# NTF Streams Architecture

### Contents

- [Streams Architecture](#streams-architecture)
- [Platform Protocols](#platform-abstraction)
    - [ComputePlatform](#ComputePlatform)
    - [ComputeService](#ComputeService)
    - [ComputeDevice](#ComputeDevice)
    - [DeviceArray](#DeviceArray)
    - [DeviceStream](#DeviceStream)
    - [StreamEvent](#StreamEvent)

## Overview
This document describes what streams are and how they work in detail, along with the platform protocols.

## Platform Abstraction
NTF defines a set of class protocols for platform abstraction. They encapsulate functionality to allow run time selection of compute services (cpu, cuda, metal, etc.) and devices to enable application portability without recoding or recompiling.
```swift
ComputePlatform         // local, remote
  services[]
    ComputeService      // (cpu, cuda, amd, tpu, ...)
      devices[]
        ComputeDevice   // (gpu:0, gpu:1, ...)
          DeviceArray
          DeviceStream
        StreamEvent
```
# Streams Architecture
## Device Stream
A device stream is an interface to a set of tensor operations that are executed via a serial asynchronous FIFO queue. Each compute service will have unique hardware specific  `DeviceStream`  implementations. The `CpuStream` class will be optimized for the host cpu, and the CudaStream implementation efficiently wraps Cuda streams.

![Device Stream](https://github.com/ewconnell/NetlibTF/blob/master/Docs/Diagrams/DeviceStreamQueue.png)

### DeviceStream Functions and Drivers 
Device stream implementations conform to the _StreamIntrinsicsProtocol_ which queues a broad set of basic tensor functions. Additional protocols with higher level functions are also adopted by the stream. These higher level functions are aggregates whose default implementation use the intrinsics layer to complete the task. However, a stream implementer has the opportinity to directly implement any level of function optimized for the target device. This approach allows the first version of a new device driver to be implemented correctly, because all functions can start with the default cpu implementation. Later the driver developer can substitute hardware specific implementations.

### StreamEvent
A stream event is a synchronization object with barrier semantics used to synchronize streams with the host thread or other streams in a multi stream application. Stream synchronozation is handled automatically, so the application developer has no need to interact with stream events directly.

The cost measured on a 2.3 ghz i5 MacBook Pro is about 1.5 million events can be created and destroyed per second. This seems reasonably cheap, with 1 event created each time a tensor is accessed by a different stream, or when the application thread request access to the data.

![Queued StreamEvents](https://github.com/ewconnell/NetlibTF/blob/master/Docs/Diagrams/StreamRecordedEvents.png)

## Tensors and Streams
A tensor is an n-dimensional data array used by stream functions as input and output. A stream function executes on the associated device, which might exist in a unified memory address space with the application, or in a discrete address space. Memory management and synchronization are transparently managed for the user.

![Tensor Structure](https://github.com/ewconnell/NetlibTF/blob/master/Docs/Diagrams/TensorStructure.png)

### Tensor Initialization
Shaped tensors such as Vector and Matrix and just constrained variations of _TensorView_ which is a generalized n-dimensional tensor. Almost all functions will operate on data objects that generically conform to _TensorView_ and not one of the shaped types. The shaped types are for the purpose of application clarity, and optimized indexing in the application space. The user is unaware of the underlying _TensorArray_ and _DeviceArray_ objects. Shaped tensors provide shape specific _init_ functions, that create and initialize a _TensorArray_ then call a uniform view initializer. TensorViews are structs, and TensorArrays are class objects shared by views.

A TensorArray can be created:
- empty with no space
- specifying size without initial values. This variation is lazily allocated the first time data access is attempted.
- specifying size with initial values

## Tensor Synchronization
Tensors can be freely used on multiple streams and in multiple application threads. However, the application writer needs to be aware that multiple simultaneous writers will cause copy-on-write mutation. The logging diagnostics can be set to report tensor mutation to simplify checking that a design is working as intended.

### Tensor Sync Process
Each time write access to a _TensorArray_ is obtained, the `lastMutatingStream` member is checked to see if it matches the requesting stream. If they do not match, then the streams are synchronized at this point. The private _TensorView_ helper function `synchronize` is called to do the work.
```swift
func synchronize(stream lastStream: DeviceStream?, with nextStream: DeviceStream)
```
An event is created and recorded on the `lastStream` and a wait is queued on the `nextStream` to ensure continuity.

# Platform protocols
## ComputePlatform
A compute platform represents the root for managing all services, devices, and streams on a platform. There is one local instance per process, and possibly many remote instances.
### Platform Class
NTF defines the _Platform_ singleton class that adopts the _ComputePlatform_ and _LocalPlatform_ protocols. _ComputePlatform_ is a class protocol, and _LocalPlatform_ is a default implementation. Additional default implementations will be added for remoting.
```swift
public protocol ComputePlatform: DeviceErrorHandling, ObjectTracking, Logger {
    /// global shared instance
    static var local: Platform { get }
    /// the root log
    static var log: Log { get set }
    
    // instance members
    /// a device automatically selected based on service priority
    var defaultDevice: ComputeDevice { get }
    /// ordered list of device ids specifying the order for auto selection
    var deviceIdPriority: [Int] { get set }
    /// the platform id. Usually zero, but can be assigned in case a higher
    /// level object (e.g. cluster) will maintain a platform collection
    var id: Int { get set }
    /// location of dynamically loaded service modules
    var serviceModuleDirectory: URL { get set }
    /// ordered list of service names specifying the order for auto selection
    var servicePriority: [String] { get set }
    /// a dynamically loaded collection of available compute services.
    /// The "cpu" service will always be available
    var services: [String : ComputeService] { get }
    
    //--------------------------------------------------------------------------
    /// createStream will try to match the requested service name and
    /// device id returning substitutions if needed to fulfill the request
    ///
    /// Parameters
    /// - Parameter deviceId: (0, 1, 2, ...)
    ///   If the specified id is greater than the number of available devices,
    ///   then id % available will be used.
    /// - Parameter serviceName: (cpu, cuda, tpu, ...)
    ///   If no service name is specified, then the default is used.
    /// - Parameter name: a text label assigned to the stream for logging
    func createStream(deviceId: Int,
                      serviceName: String?,
                      name: String) -> DeviceStream
    
    //--------------------------------------------------------------------------
    /// requestDevices
    /// - Parameter serviceName: the service to allocate the device from.
    /// - Parameter deviceId: selected device id
    /// - Returns: the requested device from the requested service
    ///   substituting if needed based on `servicePriority`
    ///   and `deviceIdPriority`
    func requestDevice(serviceName: String, deviceId: Int) -> ComputeDevice?
}
```
The _LocalPlatform_ protocol adds the remote open function
```swift
/// The default ComputePlatform implementation for a local host
public protocol LocalPlatform : ComputePlatform {
    static func open(platform url: URL) throws -> ComputePlatform
    /// a platform wide unique device id obtained during initialization
    static var nextUniqueDeviceId: Int { get }
    /// a platform wide unique stream id obtained during initialization
    static var nextUniqueStreamId: Int { get }
}
```
## ComputeService
A compute service implements the _ComputeService_ protocol and is used to enumerate available devices.
```swift
public protocol ComputeService: ObjectTracking, Logger, DeviceErrorHandling {
    /// a collection of available devices
    var devices: [ComputeDevice] { get }
    /// the service id
    var id: Int { get }
    /// the service name used for `servicePriority` and logging
    var name: String { get }
    /// the platform this service instance belongs to
    var platform: ComputePlatform! { get }
    /// the default maximum amount of time allowed for an operation to complete
    /// this is inherited by devices and streams when they are created
    var timeout: TimeInterval { get set }

    /// required initializer to support dynamically loaded services
    init(platform: ComputePlatform, 
         id: Int,
         logInfo: LogInfo, 
         name: String?) throws
}
```
### ComputeDevice
A compute device implements the _ComputeDevice_ protocol and is used to query device attributes, and create resources such as device arrays and streams.
```swift
public protocol ComputeDevice: ObjectTracking, Logger, DeviceErrorHandling {
    /// a dictionary of device specific attributes describing the device
    var attributes: [String: String] { get }
    /// the amount of free memory currently available on the device
    var availableMemory: UInt64 { get }
    /// a key to lookup device array replicas
    var deviceArrayReplicaKey: Int { get }
    /// the id of the device for example gpu:0
    var id: Int { get }
    /// the maximum number of threads supported per block
    var maxThreadsPerBlock: Int { get }
    /// the name of the device
    var name: String { get }
    /// the service this device belongs to
    var service: ComputeService! { get }
    /// the maximum amount of time allowed for an operation to complete
    var timeout: TimeInterval { get set }
    /// the type of memory addressing this device uses
    var memoryAddressing: MemoryAddressing { get }
    /// current percent of the device utilized
    var utilization: Float { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    func createArray(count: Int) throws -> DeviceArray
    /// creates a device array from a uma buffer
    func createMutableReferenceArray(buffer: UnsafeMutableRawBufferPointer)
        -> DeviceArray
    /// creates a device array from a uma buffer
    func createReferenceArray(buffer: UnsafeRawBufferPointer) -> DeviceArray
    /// creates a named command stream for this device
    func createStream(name: String) -> DeviceStream
}

public enum MemoryAddressing { case unified, discreet }
```
## DeviceStream
A device stream is an abstraction for an asynchronous command queue that implements the _DeviceStream_ protocol. It is used to schedule and synchronize computations. The protocol function implementations are service API specific and optimized.
```swift
public protocol DeviceStream:
    ObjectTracking,
    Logger,
    DeviceErrorHandling,
    StreamIntrinsicsProtocol,
    StreamGradientsProtocol
{
    //--------------------------------------------------------------------------
    /// options to use when creating stream events
    var defaultStreamEventOptions: StreamEventOptions { get }
    /// the device the stream is associated with
    var device: ComputeDevice { get }
    /// if `true` the stream will execute functions synchronous with the app
    /// it is `false` by default and used for debugging
    var executeSynchronously: Bool { get set }
    /// a unique id used to identify the stream
    var id: Int { get }
    /// a name used to identify the stream
    var name: String { get }
    /// the maximum time to wait for an operation to complete
    /// a value of 0 (default) will wait forever
    var timeout: TimeInterval { get set }
    
    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a StreamEvent
    func createEvent(options: StreamEventOptions) throws -> StreamEvent
    /// queues a stream event op. When executed the event is signaled
    @discardableResult
    func record(event: StreamEvent) throws -> StreamEvent
    /// records an op on the stream that will perform a stream blocking wait
    /// when it is processed
    func wait(for event: StreamEvent) throws
    /// blocks the calling thread until the stream queue has completed all work
    func waitUntilStreamIsComplete() throws

    //--------------------------------------------------------------------------
    // debugging functions
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the stream by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    func simulateWork<T>(x: T, timePerElement: TimeInterval, result: inout T)
        where T: TensorView
    /// causes the stream to sleep for the specified interval for testing
    func delayStream(atLeast interval: TimeInterval)
    /// for unit testing. It's part of the class protocol so that remote
    /// streams throw the error remotely.
    func throwTestError()
}
```
### DeviceArray
A device array implements the _DeviceArray_ protocol and is an abstraction for a contiguous array of bytes on a device. 
```swift
public protocol DeviceArray: ObjectTracking {
    /// the device that created this array
    var device: ComputeDevice { get }
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }

    //-------------------------------------
    /// asynchronously copies the contents of another device array
    func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
    /// asynchronously copies the contents of an app memory buffer
    func copyAsync(from buffer: UnsafeRawBufferPointer,
                   using stream: DeviceStream) throws
    /// copies the contents to an app memory buffer asynchronously
    func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                   using stream: DeviceStream) throws
    /// clears the array to zero
    func zero(using stream: DeviceStream) throws
}
```
### StreamEvent
A stream event implements the _StreamEvent_ protocol and is used to synchronize device streams.
A stream event is a synchronization object with barrier semantics, which is:
- created by a `DeviceStream`
- recorded on a stream to create a barrier
- waited on by one or more threads for group synchronization

```swift
public protocol StreamEvent: ObjectTracking {
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    /// the last time the event was recorded
    var recordedTime: Date? { get set }
    /// measure elapsed time since another event
    func elapsedTime(since other: StreamEvent) -> TimeInterval?
    /// will block the caller until the timeout has elapsed if one
    /// was specified during init, otherwise it will block forever
    func wait() throws
}
```
