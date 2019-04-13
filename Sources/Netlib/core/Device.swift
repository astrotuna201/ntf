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
    /// the maximum amount of time allowed for an operation to complete
    var timeout: TimeInterval? { get set }
    /// the type of memory addressing this device uses
    var memoryAddressing: MemoryAddressing { get }
    /// current percent of the device utilized
    var utilization: Float { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    func createArray(count: Int) throws -> DeviceArray
    /// creates a named command stream for this device
    func createStream(name: String) throws -> DeviceStream
}

public enum MemoryAddressing { case unified, discreet }

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
    func copyAsync(from buffer: UnsafeRawBufferPointer,
                   using stream: DeviceStream) throws
    /// copies the contents to a memory buffer synchronously
    func copy(to buffer: UnsafeMutableRawBufferPointer,
              using stream: DeviceStream) throws
    /// copies the contents to a memory buffer asynchronously
    func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                   using stream: DeviceStream) throws
}

//==============================================================================
// StreamEvent
/// Stream events are queued to enable stream synchronization
public protocol StreamEvent: ObjectTracking, Logging {
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    
    // TODO: consider adding time outs for failed remote events
    init(options: StreamEventOptions) throws
}

public struct StreamEventOptions: OptionSet {
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    public static let hostSync     = StreamEventOptions(rawValue: 1 << 0)
    public static let timing       = StreamEventOptions(rawValue: 1 << 1)
    public static let interprocess = StreamEventOptions(rawValue: 1 << 2)
}

public enum StreamEventError: Error {
    case timedOut
}
