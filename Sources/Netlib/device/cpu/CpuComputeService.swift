//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// CpuComputeService
public class CpuComputeService : LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var _deviceErrorHandler: DeviceErrorHandler! = nil
    public var _lastError: Error? = nil
    public var errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String?) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "cpu"
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)

        // add cpu device
        devices.append(CpuDevice(service: self, deviceId: 0,
                                 logInfo: logInfo.flat("cpu:0"),
                                 memoryAddressing: .unified))
        
        // pointer to instance error handler function
        _deviceErrorHandler = defaultDeviceErrorHandler(error:)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}

//==============================================================================
/// CpuUnitTestComputeService
/// This is used for unit testing only
public class CpuUnitTestComputeService : LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var _deviceErrorHandler: DeviceErrorHandler! = nil
    public var _lastError: Error? = nil
    public var errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    
    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String?) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "cpuUnitTest"
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)
        
        // add cpu device
        devices.append(CpuDevice(service: self, deviceId: 0,
                                 logInfo: logInfo.flat("cpu:0"),
                                 memoryAddressing: .unified))
        
        // add two discreet versions for unit testing
        // TODO is there a better solution for testing
        devices.append(CpuDevice(service: self, deviceId: 1,
                                 logInfo: logInfo.flat("cpu:1"),
                                 memoryAddressing: .discreet))
        
        devices.append(CpuDevice(service: self, deviceId: 2,
                                 logInfo: logInfo.flat("cpu:2"),
                                 memoryAddressing: .discreet))
        
        // pointer to instance error handler function
        _deviceErrorHandler = defaultDeviceErrorHandler(error:)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
