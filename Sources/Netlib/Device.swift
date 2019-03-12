//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

//==============================================================================
// Platform
public protocol Platform : ObjectTracking, Logging {
    var devices: [Device] { get }
}

//==============================================================================
// Device
//    This specifies the compute device interface
public protocol Device : ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// a dictionary of device specific attributes describing the device
    var attributes: [String:String] { get }
    /// the amount of free memory currently available on the device
    var availableMemory: UInt64 { get }
    /// the id of the device for example gpu:0
    var id: Int { get }
    /// the maximum number of threads supported per block
    var maxThreadsPerBlock: Int { get }
    /// the name of the device
    var name: String { get }
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
    func createStream(label: String) throws -> DeviceStream
    /// selects this device to support legacy CUDA driver libraries
    func select() throws
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray : ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// the device where this array is allocated
    var device: Device { get }
    /// a pointer to the memory on the device
    var data: UnsafeMutableRawPointer { get }
    /// the size of the device memory in bytes
    var count: Int { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }
    
    //-------------------------------------
    // functions
    /// clears the array to zero
    func zero(using stream: DeviceStream) throws
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
public protocol StreamEvent : ObjectTracking {
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
public protocol DeviceStream : ObjectTracking, Logging {
    //-------------------------------------
    // properties
    /// the device the stream is associated with
    var device: Device { get }
    /// a unique id used to identify the stream
    var id: Int { get }
    /// a label used to identify the stream
    var label: String { get }
    
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
    /// blocks caller until the event has occurred on this stream, then recorded
    /// and occurred on the other stream
    func sync(with other: DeviceStream, event: StreamEvent) throws
    /// blocks caller until the event has occurred
    func wait(for event: StreamEvent) throws
}
