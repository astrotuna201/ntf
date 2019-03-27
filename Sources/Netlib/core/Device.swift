//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// ComputeService
public protocol ComputeService: ObjectTracking, Logging {
    init(logging: LogInfo) throws
    var devices: [ComputeDevice] { get }
    var id: Int { get set }
    var name: String { get }
}

//==============================================================================
// ComputeDevice
//    This specifies the compute device interface
public protocol ComputeDevice: ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// a dictionary of device specific attributes describing the device
    var attributes: [String: String] { get }
    /// the amount of free memory currently available on the device
    var availableMemory: UInt64 { get }
    /// the id of the device for example gpu:0
    var id: Int { get }
    /// the maximum number of threads supported per block
    var maxThreadsPerBlock: Int { get }
    /// the name of the device
    var name: String { get }
    /// the service this device belongs to
    var service: ComputeService! { get }
    /// is `true` if the device is configured to use unified memory addressing
    /// with the host CPU
    var usesUnifiedAddressing: Bool { get }
    /// current percent of the device utilized
    var utilization: Float { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    func createArray(count: Int) throws -> DeviceArray
    /// creates a named command stream for this device
    func createStream(name: String) throws -> DeviceStream
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// the device where this array is allocated
    var device: ComputeDevice { get }
    /// a pointer to the memory on the device
    var data: UnsafeMutableRawPointer { get }
    /// the size of the device memory in bytes
    var count: Int { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }

    //-------------------------------------
    // functions
    /// clears the array to zero
    func zero(using stream: DeviceStream?) throws
    /// asynchronously copies the contents of another device array
    func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
    /// asynchronously copies the contents of a memory buffer
    func copyAsync(from buffer: UnsafeBufferPointer<UInt8>,
                   using stream: DeviceStream) throws
    /// copies the contents to a memory buffer synchronously
    func copy(to buffer: UnsafeMutableBufferPointer<UInt8>,
              using stream: DeviceStream) throws
    /// copies the contents to a memory buffer asynchronously
    func copyAsync(to buffer: UnsafeMutableBufferPointer<UInt8>,
                   using stream: DeviceStream) throws
}

//==============================================================================
// StreamEvent
/// Stream events are queued to enable stream synchronization
public protocol StreamEvent: ObjectTracking, Logging {
    /// is `true` if the even has occurred
    var occurred: Bool { get }

    init(options: StreamEventOptions) throws
}

public struct StreamEventOptions: OptionSet {
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    public static let hostSync     = StreamEventOptions(rawValue: 1 << 0)
    public static let timing       = StreamEventOptions(rawValue: 1 << 1)
    public static let interProcess = StreamEventOptions(rawValue: 1 << 2)
}

//==============================================================================
// DeviceStream
/// A device stream is an asynchronous queue of commands executed on
/// the associated device
public protocol DeviceStream: ObjectTracking, Logging {
    /// the device the stream is associated with
    var device: ComputeDevice { get }
    /// a unique id used to identify the stream
    var id: Int { get }
    /// a name used to identify the stream
    var name: String { get }

    //-------------------------------------
    /// execute function
    /// - Parameter functionId: id of the function to execute
    /// - Parameter instanceId: function instance id for associated resources
    /// - Parameter parameters: parameters structure to serialize
    ///
    func execute<T>(functionId: UUID, with parameters: T) throws
    
    //-------------------------------------
    /// setup a function instance and assoicated resources for
    /// repetative execution (e.g. convolution). The instanceId can
    /// susequently be used with the `execute(functionId:` function
    ///
    /// - Parameter functionId: The function class to create
    /// - Parameter instanceId: The function instance id
    /// - Parameter parameters: The configuration parameters
    ///
    func setup<T>(functionId: UUID, instanceId: UUID, with parameters: T) throws

    //-------------------------------------
    /// release function instance created by createFunctionInstance
    ///
    /// - Parameter instance: The id of the function instance to release
    ///
    func release(instanceId: UUID) throws
    
    //-------------------------------------
    // synchronization
    /// blocks the calling thread until the stream queue is empty
    func blockCallerUntilComplete() throws
    /// creates a StreamEvent
    func createEvent(options: StreamEventOptions) throws -> StreamEvent
    /// creates an artificial delay used to simulate work for debugging
    func debugDelay(seconds: Double) throws
    /// queues a stream event
    func record(event: StreamEvent) throws -> StreamEvent
    /// blocks caller until the event has occurred on this stream,
    /// then recorded and occurred on the other stream
    func sync(with other: DeviceStream, event: StreamEvent) throws
    /// blocks caller until the event has occurred
    func wait(for event: StreamEvent) throws
}
