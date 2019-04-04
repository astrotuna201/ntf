//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// CpuStreamEvent
final public class CpuStreamEvent : StreamEvent {
    // properties
    private let semaphore = DispatchSemaphore(value: 0)
    public private (set) var trackingId = 0
    public var occurred: Bool = false
    public var logging: LogInfo? = nil

    //--------------------------------------------------------------------------
    // initializers
    public required init(options: StreamEventOptions) {
        trackingId = ObjectTracker.global
            .register(self, namePath: logging?.namePath)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // functions
    /// These are not exposed through the protocol because they should only
    /// be manipulated via the DeviceStream protocol
    public func signal() {
        occurred = true
        semaphore.signal()
    }
    
    public func wait(until timeout: TimeInterval?) throws {
        if let timeout = timeout {
            let waitUntil = DispatchWallTime.now() + (timeout * 1000000)
            if semaphore.wait(wallTimeout: waitUntil) == .timedOut {
                throw StreamEventError.timedOut
            }
        } else {
            semaphore.wait()
        }
    }
}
