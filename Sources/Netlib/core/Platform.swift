//******************************************************************************
//  Created by Edward Connell on 8/20/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  Platform (local)
//	  services[]
//	    ComputeService (cpu, cuda, amd, tpu, ...)
//		  devices[]
//		    ComputeDevice (gpu:0, gpu:1, ...)
//            DeviceArray
//		      DeviceStream
//            StreamEvent
//
import Foundation


//==============================================================================
// Platform
/// The root service to enumerate and select compute services and devices
final public class Platform: ComputePlatform {
    /// a device automatically selected based on service priority
    public lazy var defaultDevice: ComputeDevice = { selectDefaultDevice() }()
    /// the default number of devices to spread a set of streams across
    /// a value of -1 specifies all available devices within the service
    public var defaultDevicesToAllocate = -1
    /// ordered list of device ids specifying the order for auto selection
    public var deviceIdPriority: [Int] = [0]
    /// global shared instance
    public static let local = Platform()
    /// location of dynamically loaded service modules
    public var serviceModuleDirectory: URL = URL(fileURLWithPath: "TODO")
    /// ordered list of service names specifying the order for auto selection
    public var servicePriority = ["cuda", "cpu"]
    /// object tracking id
    public private(set) var trackingId = 0
    /// logging information
    public var logging: LogInfo?

    //--------------------------------------------------------------------------
    /// log
    /// the caller can specify a root log which will be inherited by the
    /// device stream hierarchy, but can be overriden at any point down
    /// the tree
    public var log: Log {
        get { return logging!.log }
        set { logging!.log = newValue }
    }
    
    //--------------------------------------------------------------------------
    /// a stream created on the default device
    public private(set) static var defaultStream: DeviceStream = {
        do {
            return try local.defaultDevice
                .createStream(name: "Platform.defaultStream")
        } catch {
            // this should never fail
            local.writeLog(String(describing: error))
            fatalError()
        }
    }()

    //--------------------------------------------------------------------------
    /// collection of registered compute services (cpu, cuda, ...)
    /// loading and enumerating services is expensive and invariant, so
    /// we only want to do it once per process and share it across all
    /// Platform instances.
    public lazy var services: [String : ComputeService] = {
        if Platform._services == nil { Platform._services = loadServices() }
        return Platform._services!
    }()
    
    /// this stores the global services collection initialized by getServices
    private static var _services: [String: ComputeService]?

    //--------------------------------------------------------------------------
    // initializers
    /// `init` is private because this is a singleton. Use the `local` static
    /// member to access the global instance.
    private init() {
        let namePath = String(describing: Platform.self)
        let info = LogInfo(log: Log(), logLevel: .error,
                           namePath: namePath, nestingLevel: 0)
        self.logging = info
    }
    
    //--------------------------------------------------------------------------
    // loadServices
    // dynamically loads ComputeService bundles/dylib from the
    // `serviceModuleDirectory` and adds them to the `services` list
    private func loadServices() -> [String: ComputeService] {
        var loadedServices = [String: ComputeService]()
        do {
            func addService(_ service: ComputeService) {
                service.id = loadedServices.count
                loadedServices[service.name] = service
            }
            
            // add cpu service by default
            try addService(CpuComputeService(logging: logging!))
            //            #if os(Linux)
            //            try add(service: CudaComputeService(logging: logging))
            //            #endif
            //-------------------------------------
            // dynamically load services
            for bundle in Platform.plugInBundles {
                try bundle.loadAndReturnError()
                //            var unloadBundle = false
                
                if let serviceType =
                    bundle.principalClass as? ComputeService.Type {
                    
                    // create the service
                    let service = try serviceType.init(logging: logging!)
                    
                    if willLog(level: .diagnostic) {
                        diagnostic(
                            "Loaded compute service '\(service.name)'." +
                            " ComputeDevice count = \(service.devices.count)",
                            categories: .initialize)
                    }
                    
                    if service.devices.count > 0 {
                        // add plugin service
                        addService(service)
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
        return loadedServices
    }
    
    //--------------------------------------------------------------------------
    // plugInBundles
    // an array of the dynamically installed bundles
    public static var plugInBundles: [Bundle] = {
        var bundles = [Bundle]()
        if let dir = Bundle.main.builtInPlugInsPath {
            let paths = Bundle.paths(forResourcesOfType: "bundle",
                                     inDirectory: dir)
            for path in paths {
                bundles.append(Bundle(url: URL(fileURLWithPath: path))!)
            }
        }
        return bundles
    }()

    //--------------------------------------------------------------------------
    // selectDefaultDevice
    // selects a ComputeDevice based on `servicePriority` and
    // `deviceIdPriority`. It is guaranteed that at least one device like
    // the cpu is available
    private func selectDefaultDevice() -> ComputeDevice {
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
        if willLog(level: .status) {
            writeLog("default device: [\(device.service.name)] \(device.name)",
                level: .status)
        }
        return device
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
    public func createStreams(name: String = "stream",
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
    public func requestDevices(deviceIds: [Int],
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
    /// - Parameter url: the location of the remote platform
    /// - Returns: a reference to the remote platform, which can be used
    ///   to query resources and create remote streams.
    public static func open(platform url: URL) throws -> ComputePlatform {
        fatalError("not implemented yet")
    }
} // ComputePlatform
