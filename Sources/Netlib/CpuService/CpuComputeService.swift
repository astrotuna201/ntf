//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public class CpuComputeService : ComputeService {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public var devices = [ComputeDevice]()
    public var id = 0
    public var logging: LogInfo?
    public let name = "cpu"

    //--------------------------------------------------------------------------
    // initializers
    public required init(logging: LogInfo) throws    {
        self.logging = logging
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)

        // add cpu device
        devices.append(CpuDevice(service: self, deviceId: id,
                                 logging: logging.child("cpu"),
                                 memoryAddressing: .unified))
        
        // add a discreet version for unit testing
        // TODO is there a better solution for testing
        devices.append(CpuDevice(service: self, deviceId: id,
                                 logging: logging.child("discreetmemorycpu"),
                                 memoryAddressing: .discreet))
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
