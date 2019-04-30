//******************************************************************************
//  Created by Edward Connell on 3/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  ComputePlatform
//      services[]
//        ComputeService (cpu, cuda, amd, tpu, ...)
//          devices[]
//            ComputeDevice (gpu:0, gpu:1, ...)
//            DeviceArray
//              DeviceStream
//            StreamEvent
//
import Foundation

//==============================================================================
/// ComputePlatform
/// this represents the root for managing all services, devices, and streams
/// on a platform. There is one local instance per process, and possibly
/// many remote instances.
public protocol ComputePlatform:
    DeviceErrorHandling,
    ObjectTracking,
    Logger
{
    //--------------------------------------------------------------------------
    // class members
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

//==============================================================================
/// ComputeService
/// a compute service represents category of installed devices on the platform,
/// such as (cpu, cuda, tpu, ...)
public protocol ComputeService: ObjectTracking, Logger, DeviceErrorHandling {
    /// a collection of available devices
    var devices: [ComputeDevice] { get }
    /// the service id
    var id: Int { get }
    /// the service name used for `servicePriority` and logging
    var name: String { get }
    /// the platform this service belongs to
    var platform: ComputePlatform! { get }
    /// required initializer to support dynamically loaded services
    init(platform: ComputePlatform, id: Int,
         logInfo: LogInfo, name: String?) throws
}

//==============================================================================
/// LocalComputeService
public protocol LocalComputeService: ComputeService { }

public extension LocalComputeService {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            platform.handleDevice(error: error)
        }
    }
}

//==============================================================================
/// ComputeDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ComputeDevice: ObjectTracking, Logger, DeviceErrorHandling {
    //-------------------------------------
    // properties
    /// a dictionary of device specific attributes describing the device
    var attributes: [String: String] { get }
    /// the amount of free memory currently available on the device
    var availableMemory: UInt64 { get }
    /// a key to lookup device array replicas
    var deviceArrayReplicaKey: DeviceArrayReplicaKey { get }
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
    /// creates a device array from a uma buffer.
    func createReferenceArray(buffer: UnsafeRawBufferPointer) -> DeviceArray
    /// creates a device array from a uma buffer.
    func createMutableReferenceArray(buffer: UnsafeMutableRawBufferPointer)
        -> DeviceArray
    /// creates a named command stream for this device
    func createStream(name: String) -> DeviceStream
}

public enum MemoryAddressing { case unified, discreet }

//==============================================================================
/// DeviceArrayReplicaKey
public struct DeviceArrayReplicaKey: Hashable {
    let platformId: UInt8
    let serviceId: UInt8
    let deviceId: UInt8
    
    public init(platformId: Int, serviceId: Int, deviceId: Int) {
        self.platformId = UInt8(platformId)
        self.serviceId = UInt8(serviceId)
        self.deviceId = UInt8(deviceId)
    }
}

//==============================================================================
/// LocalComputeDevice
public protocol LocalComputeDevice: ComputeDevice { }

public extension LocalComputeDevice {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            service.handleDevice(error: error)
        }
    }
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking {
    //-------------------------------------
    // properties
    /// the device where this array is allocated
    var device: ComputeDevice { get }
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }

    //-------------------------------------
    /// clears the array to zero
    func zero(using stream: DeviceStream) throws
    /// asynchronously copies the contents of another device array
    func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
    /// asynchronously copies the contents of an app memory buffer
    func copyAsync(from buffer: UnsafeRawBufferPointer,
                   using stream: DeviceStream) throws
    /// copies the contents to an app memory buffer asynchronously
    func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                   using stream: DeviceStream) throws
}

//==============================================================================
/// StreamEvent
/// A stream event is a barrier synchronization object that is
/// - created by a `DeviceStream`
/// - recorded on the stream to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol StreamEvent: ObjectTracking {
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get set }
    /// the stream that created this event
    var stream: DeviceStream { get }

    /// signals that the event has occurred
    func signal()
    /// will block the caller until the timeout has elapsed, or
    /// if `timeout` is 0 it will wait forever
    func blockingWait(for timeout: TimeInterval) throws
}

public extension StreamEvent {
    func blockingWait() throws {
        return try blockingWait(for: 0)
    }
}

public struct StreamEventOptions: OptionSet {
    public init() { self.rawValue = 0 }
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    public static let hostSync     = StreamEventOptions(rawValue: 1 << 0)
    public static let timing       = StreamEventOptions(rawValue: 1 << 1)
    public static let interprocess = StreamEventOptions(rawValue: 1 << 2)
}

public enum StreamEventError: Error {
    case timedOut
}
