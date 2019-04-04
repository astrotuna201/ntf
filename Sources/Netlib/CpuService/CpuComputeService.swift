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
        
        // add a device for each logical core
        let processInfo = ProcessInfo()
        for id in 0..<processInfo.activeProcessorCount {
            let device = CpuDevice(logging: logging.child("cpu:\(id)"),
                                   service: self, deviceId: id)
            devices.append(device)
        }
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
