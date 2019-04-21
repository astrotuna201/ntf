//******************************************************************************
//  Created by Edward Connell on 8/20/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// LocalPlatform
/// The default ComputePlatform implementation for a local host
public protocol LocalPlatform : ComputePlatform {
    /// the global services collection
    static var _services: [String: ComputeService]? { get set }
}

public extension LocalPlatform {
    //--------------------------------------------------------------------------
    /// log
    /// the caller can specify a root log which will be inherited by the
    /// device stream hierarchy, but can be overriden at any point down
    /// the tree
    static var log: Log {
        get { return Platform.local.logInfo.log }
        set { Platform.local.logInfo.log = newValue }
    }
    
    //--------------------------------------------------------------------------
    /// defaultDeviceErrorHandler
    func defaultDeviceErrorHandler(error: DeviceError) {
        
    }

    //--------------------------------------------------------------------------
    // loadServices
    // dynamically loads ComputeService bundles/dylib from the
    // `serviceModuleDirectory` and adds them to the `services` list
    func loadServices() {
        guard Platform._services == nil else { return }
        
        var loadedServices = [String: ComputeService]()
        do {
            // add required cpu service
            let cpuService = try CpuComputeService(id: loadedServices.count,
                                                   logInfo: logInfo, name: nil)
            loadedServices[cpuService.name] = cpuService
            
            // add cpu unit test service
            let cpuUnitTestService =
                try CpuUnitTestComputeService(id: loadedServices.count,
                                              logInfo: logInfo,
                                              name: "cpuUnitTest")
            loadedServices[cpuUnitTestService.name] = cpuUnitTestService
            
            //            #if os(Linux)
            //            try add(service: CudaComputeService(logging: logging))
            //            #endif
            //-------------------------------------
            // dynamically load installed services
            let bundles = getPlugInBundles()
            for bundle in bundles {
                try bundle.loadAndReturnError()
                //            var unloadBundle = false
                
                if let serviceType =
                    bundle.principalClass as? ComputeService.Type {
                    
                    // create the service
                    let service =
                        try serviceType.init(id: loadedServices.count,
                                             logInfo: logInfo, name: nil)
                    
                    if willLog(level: .diagnostic) {
                        diagnostic(
                            "Loaded compute service '\(service.name)'." +
                            " ComputeDevice count = \(service.devices.count)",
                            categories: .initialize)
                    }
                    
                    if service.devices.count > 0 {
                        // add plugin service
                        loadedServices[service.name] = service
                    } else {
                        writeLog("Compute service '\(service.name)' " +
                            "successfully loaded, but reported devices = 0, " +
                            "so service is unavailable", level: .warning)
                        //                    unloadBundle = true
                    }
                }
                // TODO: we should call bundle unload here if there were no devices
                // however simply calling bundle.load() then bundle.unload() making no
                // references to objects inside, later causes an exception in the code.
                // Very strange
                //            if unloadBundle { bundle.unload() }
            }
        } catch {
            writeLog(String(describing: error))
        }
        Platform._services = loadedServices
    }
    
    //--------------------------------------------------------------------------
    /// getPlugInBundles
    /// an array of the dynamically installed bundles
    private func getPlugInBundles() -> [Bundle] {
        if let dir = Bundle.main.builtInPlugInsPath {
            return Bundle.paths(forResourcesOfType: "bundle", inDirectory: dir)
                .map { Bundle(url: URL(fileURLWithPath: $0))! }
        } else {
            return []
        }
    }
    
    //--------------------------------------------------------------------------
    // selectDefaultDevice
    // selects a ComputeDevice based on `servicePriority` and
    // `deviceIdPriority`. It is guaranteed that at least one device like
    // the cpu is available
    func selectDefaultDevice() -> ComputeDevice {
        // try to exact match the service request
        var defaultDev: ComputeDevice?
        let requestedDevice = deviceIdPriority[0]
        for serviceName in servicePriority where defaultDev == nil {
            defaultDev = requestDevice(serviceName: serviceName,
                                       deviceId: requestedDevice,
                                       allowSubstitute: false)
        }
        
        // if the search failed, then allow substitutes
        if defaultDev == nil {
            let priority = servicePriority + ["cpu"]
            for serviceName in priority where defaultDev == nil {
                defaultDev = requestDevice(serviceName: serviceName,
                                           deviceId: 0,
                                           allowSubstitute: true)
            }
        }
        
        // we had to find at least one device like the cpu
        assert(defaultDev != nil, "There must be at least one device")
        let device = defaultDev!
        writeLog("default device: [\(device.service.name)] \(device.name)",
            level: .status)
        return device
    }
    
    //--------------------------------------------------------------------------
    /// createDefaultStream
    /// creates a stream on the default device
    static func createDefaultStream() -> DeviceStream {
        do {
            return try local.defaultDevice
                .createStream(name: "Platform.defaultStream")
        } catch {
            local.writeLog(String(describing: error))
            fatalError("unable to create the default stream")
        }
    }

    //--------------------------------------------------------------------------
    /// createStreams
    //
    /// This will try to match the requested service and device ids returning
    /// substitutes if needed.
    ///
    /// Parameters
    /// - Parameter label: The text label applied to the stream
    /// - Parameter serviceName: (cpu, gpu, tpu, ...)
    ///   If no service name is specified, then the default is used.
    /// - Parameter deviceIds: (0, 1, 2, ...)
    ///   If no ids are specified, then one stream per defaultDeviceCount
    ///   is returned.
    func createStreams(name: String = "stream",
                       serviceName: String? = nil,
                       deviceIds: [Int]? = nil) throws -> [DeviceStream]{
        
        // choose the service to select the device from
        let serviceName = serviceName ?? defaultDevice.service.name
        
        // choose how many devices to spread the streams across
        let maxDeviceCount = defaultDevicesToAllocate == -1 ?
            defaultDevice.service.devices.count :
            min(defaultDevicesToAllocate, defaultDevice.service.devices.count)
        
        // get the device ids
        let ids = deviceIds ?? [Int](0..<maxDeviceCount)
        
        // create the streams
        return try ids.map {
            let device = requestDevice(serviceName: serviceName,
                                       deviceId: $0, allowSubstitute: true)!
            return try device.createStream(name: name)
        }
    }
    
    //--------------------------------------------------------------------------
    /// requestDevices
    /// This will try to return the requested devices from the requested service
    /// substituting if needed based on `servicePriority` and `deviceIdPriority`
    ///
    func requestDevices(deviceIds: [Int],
                        serviceName: String?) -> [ComputeDevice] {
        // if no serviceName is specified then return the default
        let serviceName = serviceName ?? defaultDevice.service.name
        return deviceIds.map {
            requestDevice(serviceName: serviceName,
                          deviceId: $0, allowSubstitute: true)!
        }
    }
    
    //--------------------------------------------------------------------------
    // requestDevice
    /// This tries to satisfy the device requested, but if not available will
    /// return a suitable alternative. In the case of an invalid string, an
    /// error will be reported, but no exception will be thrown
    private func requestDevice(serviceName: String, deviceId: Int,
                               allowSubstitute: Bool) -> ComputeDevice? {
        
        if let service = services[serviceName] {
            if deviceId < service.devices.count {
                return service.devices[deviceId]
            } else if allowSubstitute {
                return service.devices[deviceId % service.devices.count]
            } else {
                return nil
            }
        } else if allowSubstitute {
            let service = services[defaultDevice.service.name]!
            return service.devices[deviceId % service.devices.count]
        } else {
            return nil
        }
    }
    
    //--------------------------------------------------------------------------
    /// open
    /// this is a placeholder. Additional parameters will be needed for
    /// credentials, timeouts, etc...
    ///
    /// - Parameter url: the location of the remote platform
    /// - Returns: a reference to the remote platform, which can be used
    ///   to query resources and create remote streams.
    static func open(platform url: URL) throws -> ComputePlatform {
        fatalError("not implemented yet")
    }

}

//==============================================================================
// Platform
/// The root object to select compute services and devices
final public class Platform: LocalPlatform {
    // properties
    public lazy var defaultDevice: ComputeDevice = { selectDefaultDevice() }()
    public private(set) static var defaultStream: DeviceStream = {
        createDefaultStream()
    }()
    public var defaultDevicesToAllocate = -1
    public var _deviceErrorHandler: DeviceErrorHandler! = nil
    public var lastDeviceError: DeviceError = .none
    public var deviceIdPriority: [Int] = [0]
    public var id: Int = 0
    public static let local = Platform()
    public var serviceModuleDirectory: URL = URL(fileURLWithPath: "TODO")
    public var servicePriority = ["cuda", "cpu"]
    public lazy var services: [String : ComputeService] = {
        loadServices()
        return Platform._services!
    }()
    public static var _services: [String: ComputeService]?
    public private(set) var trackingId = 0
    public var logInfo: LogInfo

    //--------------------------------------------------------------------------
    // initializers
    /// `init` is private because this is a singleton. Use the `local` static
    /// member to access the shared instance.
    private init() {
        // log
        logInfo = LogInfo(log: Log(), logLevel: .error,
                          namePath: String(describing: Platform.self),
                          nestingLevel: 0)
        
        // pointer to instance error handler function
        _deviceErrorHandler = defaultDeviceErrorHandler(error:)
    }
}
