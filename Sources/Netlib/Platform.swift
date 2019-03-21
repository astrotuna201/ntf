//******************************************************************************
//  Created by Edward Connell on 8/20/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  EvaluationContext (local, remote)
//    Platform (global)
//		services[]
//		  ComputeService (cpu, cuda, amd, tpu, ...)
//			devices[]
//			  ComputeDevice (gpu:0, gpu:1, ...)
//			    DeviceStream
//				DeviceArray
//
import Foundation

//==============================================================================
// Platform
/// The root service to enumerate and select compute services and devices
final public class Platform: ObjectTracking, Logging {
    //--------------------------------------------------------------------------
    // properties
    public weak var context: EvaluationContext!
    /// a device automatically selected during init based on service priority
    public lazy var defaultDevice: ComputeDevice = { selectDefaultDevice() }()
    ///
    public var defaultDeviceCount = 1
    /// ordered list of device ids specifying the order for auto selection
    public var devicePriority: [Int]?
    /// default device stream
    public lazy var defaultStream: DeviceStream = {
        do {
            return try self.defaultDevice.createStream(label: "Platform.defaultStream")
        } catch {
            writeLog(String(describing: error))
            fatalError()
        }
    }()
    /// ordered list of service names specifying the order for auto selection
    public var servicePriority = ["cuda", "cpu"]
    /// location of dynamically loaded service modules
    public var servicesLocation: URL = URL(fileURLWithPath: "TODO")

    // object tracking
    public private(set) var trackingId = 0
    public var namePath = String(describing: Platform.self)

    // logging
    public var log: Log?
    public var logLevel = LogLevel.error
    public let nestingLevel = 0

    //--------------------------------------------------------------------------
    // init
    public init(context: EvaluationContext) {
        self.context = context
        self.log = context.log
    }
    
    //--------------------------------------------------------------------------
    /// collection of registered compute services (cpu, cuda, ...)
    private static var _services: [String: ComputeService]!
    public lazy var services: [String: ComputeService] = {
        Platform.getServices(instance: self)
    }()
    
    private class func getServices(instance: Platform) -> [String: ComputeService] {
        guard Platform._services == nil else { return Platform._services }
        var _services = [String: ComputeService]()
        // helper
        func add(service: ComputeService) {
            service.id = _services.count
            _services[service.name] = service
        }
        
        do {
            // add cpu service by default
            // TODO: put back!
            //            try add(service: CpuComputeService(log: currentLog))
            //            #if os(Linux)
            //            try add(service: CudaComputeService(log: currentLog))
            //            #endif
            //-------------------------------------
            // dynamically load services
            for bundle in Platform.plugInBundles {
                try bundle.loadAndReturnError()
                //            var unloadBundle = false
                
                if let serviceType = bundle.principalClass as? ComputeService.Type {
                    // create the service
                    let service = try serviceType.init(log: instance.log)
                    
                    if instance.willLog(level: .diagnostic) {
                        instance.diagnostic(
                            "Loaded compute service '\(service.name)'." +
                            " ComputeDevice count = \(service.devices.count)",
                            categories: .setup)
                    }
                    
                    if service.devices.count > 0 {
                        // add plugin service
                        add(service: service)
                    } else {
                        if instance.willLog(level: .warning) {
                            instance.writeLog(
                                "Compute service '\(service.name)' successfully loaded, " +
                                "but reported devices = 0, so service is unavailable",
                                level: .warning)
                        }
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
            instance.writeLog(String(describing: error))
        }
        return _services
    }
    
    //--------------------------------------------------------------------------
    // plugIns TODO: move to compute service
    public static var plugInBundles: [Bundle] = {
        var bundles = [Bundle]()
        if let dir = Bundle.main.builtInPlugInsPath {
            let paths = Bundle.paths(forResourcesOfType: "bundle", inDirectory: dir)
            for path in paths {
                bundles.append(Bundle(url: URL(fileURLWithPath: path))!)
            }
        }
        return bundles
    }()

    //--------------------------------------------------------------------------
    // selectDefaultDevice
    private func selectDefaultDevice() -> ComputeDevice {
        // try to exact match the service request
        var defaultDev: ComputeDevice?
        let requestedDevice = devicePriority?[0] ?? 0
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
        if willLog(level: .status) {
            writeLog("""
                default device: \(defaultDevice.name)
                id: \(defaultDevice.service.name).\(defaultDevice.id)
                """, level: .status)
        }

        // we had to find at least one device like the cpu
        guard let device = defaultDev else { fatalError("No available devices") }
        return device
    }

    //--------------------------------------------------------------------------
    // add(service
    public func add(service: ComputeService) {
        service.id = services.count
        services[service.name] = service
    }

    //--------------------------------------------------------------------------
    // requestStreams
    //	This will try to match the requested service and device ids returning
    // substitutes if needed.
    //
    // If no service name is specified, then the default is used.
    // If no ids are specified, then one stream per defaultDeviceCount is returned
    public func requestStreams(label: String,
                               serviceName: String? = nil,
                               deviceIds: [Int]? = nil) throws -> [DeviceStream] {

        let serviceName = serviceName ?? defaultDevice.service.name
        let maxDeviceCount = min(defaultDeviceCount, defaultDevice.service.devices.count)
        let ids = deviceIds ?? [Int](0..<maxDeviceCount)

        return try ids.map {
            let device = requestDevice(serviceName: serviceName,
                    deviceId: $0, allowSubstitute: true)!
            return try device.createStream(label: label)
        }
    }

    //--------------------------------------------------------------------------
    // requestDevices
    public func requestDevices(serviceName: String?, deviceIds: [Int]) -> [ComputeDevice] {
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
    public func requestDevice(serviceName: String, deviceId: Int,
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
} // ComputePlatform
