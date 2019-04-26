//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// CpuStreamEvent
/// a stream event behaves like a barrier. The first caller to wait takes
/// the wait semaphore
final public class CpuStreamEvent : StreamEvent {
    // properties
    public var logInfo: LogInfo
    private let occurredMutex = Mutex()
    private let semaphore = DispatchSemaphore(value: 0)
    public private (set) var trackingId = 0
    private var _occurred: Bool = true
    private let name: String

    //--------------------------------------------------------------------------
    // initializers
    public required init(logInfo: LogInfo, options: StreamEventOptions) {
        self.logInfo = logInfo
        name = String(describing: CpuStreamEvent.self)
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // functions
    /// These are not exposed through the protocol because they should only
    /// be manipulated via the DeviceStream protocol
    public func signal() {
        semaphore.signal()
    }
    
    public var occurred: Bool {
        get { return occurredMutex.sync { _occurred } }
        set { occurredMutex.sync { _occurred = newValue } }
    }
    
    public func wait(until timeout: TimeInterval?) throws {
        try occurredMutex.sync {
            guard !_occurred else { return }
            diagnostic("\(name)(\(trackingId)) waiting...",
                categories: .streamSync)

            if let timeout = timeout {
                let waitUntil = DispatchWallTime.now() + (timeout * 1000000)
                if semaphore.wait(wallTimeout: waitUntil) == .timedOut {
                    diagnostic("\(name)(\(trackingId)) timed out",
                        categories: .streamSync)
                    throw StreamEventError.timedOut
                }
            } else {
                semaphore.wait()
            }

            diagnostic("\(name)(\(trackingId)) occured",
                categories: .streamSync)
            _occurred = true
        }
    }
}
