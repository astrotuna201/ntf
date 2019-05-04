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
    public private (set) var trackingId = 0
    private let access = Mutex()
    public var device: ComputeDevice
    private let semaphore = DispatchSemaphore(value: 0)
    private var _occurred: Bool = true

    //--------------------------------------------------------------------------
    // initializers
    public init(device: ComputeDevice, options: StreamEventOptions) {
        self.device = device
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)

        device.diagnostic(
            "\(createString) StreamEvent(\(trackingId)) on \(device.name)",
            categories: .streamSync)
        #endif
    }
    deinit {
        // signal if anyone was waiting
        signal()
        
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // functions
    ///
    /// These are not exposed through the protocol because they should only
    /// be manipulated via the DeviceStream protocol
    public func signal() {
        semaphore.signal()
    }
    
    public var occurred: Bool {
        get { return access.sync { _occurred } }
    }
    
    public func blockingWait(for timeout: TimeInterval) throws {
        try access.sync {
            guard !_occurred else { return }
            #if DEBUG
            device.diagnostic(
                "\(waitString) StreamEvent(\(trackingId)) on \(device.name)",
                categories: .streamSync)
            #endif
            
            if timeout > 0 {
                if semaphore.wait(timeout: .now() + timeout) == .timedOut {
                    #if DEBUG
                    device.diagnostic(
                        "StreamEvent(\(trackingId)) timed out",
                        categories: .streamSync)
                    #endif
                    throw StreamEventError.timedOut
                }
            } else {
                semaphore.wait()
            }
            
            #if DEBUG
            device.diagnostic(
                "\(signaledString) StreamEvent(\(trackingId)) on \(device.name)",
                categories: .streamSync)
            #endif
            _occurred = true
        }
    }
}
