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
    /// a stream selected based on `servicePriority` and `deviceIdPriority`
    static var defaultStream: DeviceStream { get }
    
    // instance members
    /// a device automatically selected based on service priority
    var defaultDevice: ComputeDevice { get }
    /// the default number of devices to spread a set of streams across
    /// a value of -1 specifies all available devices within the service
    var defaultDevicesToAllocate: Int { get set }
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
    /// createStreams will try to match the requested service name and
    /// device ids returning substitutions if needed to fulfill the request
    ///
    /// Parameters
    /// - Parameter name: a text label assigned to the stream for logging
    /// - Parameter serviceName: (cpu, cuda, tpu, ...)
    ///   If no service name is specified, then the default is used.
    /// - Parameter deviceIds: (0, 1, 2, ...)
    ///   If no ids are specified, then one stream per defaultDeviceCount
    ///   is returned. If device ids are specified that are greater than
    ///   the number of available devices, then id % available will be used.
    func createStreams(name: String,
                       serviceName: String?,
                       deviceIds: [Int]?) throws -> [DeviceStream]
    
    //--------------------------------------------------------------------------
    /// requestDevices
    /// - Parameter deviceIds: an array of selected device ids
    /// - Parameter serviceName: an optional service name to allocate
    ///   the devices from.
    /// - Returns: the requested devices from the requested service
    ///   substituting if needed based on `servicePriority`
    ///   and `deviceIdPriority`
    func requestDevices(deviceIds: [Int],
                        serviceName: String?) -> [ComputeDevice]
}

//==============================================================================
/// DeviceError
public enum DeviceError : Error {
    case streamError(idPath: [Int], error: Error)
    case streamInvalidArgument(idPath: [Int], message: String, aux: Error?)
    case streamTimeout(idPath: [Int], message: String)
}

public typealias DeviceErrorHandler = (DeviceError) -> Void

public protocol DeviceErrorHandling: class {
    var _deviceErrorHandler: DeviceErrorHandler! { get set }
    var _lastDeviceError: DeviceError? { get set }
    var errorMutex: Mutex { get }
}

public extension DeviceErrorHandling {
    /// use access get/set to prevent setting `nil`
    var deviceErrorHandler: DeviceErrorHandler {
        get { return _deviceErrorHandler }
        set { _deviceErrorHandler = newValue }
    }
    
    /// safe access
    var lastDeviceError: DeviceError? {
        get { return errorMutex.sync { _lastDeviceError } }
        set { errorMutex.sync { _lastDeviceError = newValue } }
    }
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
    /// required initializer to support dynamiclly loaded services
    init(platform: ComputePlatform, id: Int,
         logInfo: LogInfo, name: String?) throws
}

//==============================================================================
/// LocalComputeService
public protocol LocalComputeService: ComputeService { }

public extension LocalComputeService {
    //--------------------------------------------------------------------------
    /// defaultDeviceErrorHandler
    func defaultDeviceErrorHandler(error: DeviceError) {
        platform.deviceErrorHandler(error)
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
/// LocalComputeDevice
public protocol LocalComputeDevice: ComputeDevice { }

public extension LocalComputeDevice {
    //--------------------------------------------------------------------------
    /// defaultDeviceErrorHandler
    func defaultDeviceErrorHandler(error: DeviceError) {
        service.deviceErrorHandler(error)
    }
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking, Logger {
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
    /// clears the array to zero
    func zero(using stream: DeviceStream?) throws
    /// asynchronously copies the contents of another device array
    func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
    /// asynchronously copies the contents of an app memory buffer
    func copyAsync(from buffer: UnsafeRawBufferPointer,
                   using stream: DeviceStream) throws
    /// copies the contents to an app memory buffer synchronously
    func copy(to buffer: UnsafeMutableRawBufferPointer,
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
public protocol StreamEvent: ObjectTracking, Logger {
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    
    // TODO: consider adding time outs for failed remote events
    init(logInfo: LogInfo, options: StreamEventOptions) throws
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
