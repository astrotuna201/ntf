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
    private let barrier = Mutex()
    public  var device: ComputeDevice
    private let occurredAccess = Mutex()
    private var _occurred: Bool = true
    private let semaphore = DispatchSemaphore(value: 0)

    //--------------------------------------------------------------------------
    // initializers
    public init(device: ComputeDevice, options: StreamEventOptions) {
        self.device = device
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)

        diagnostic(
            "\(createString) StreamEvent(\(trackingId)) on \(device.name)",
            categories: .streamSync)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // deinit
    deinit {
        // signal if anyone was waiting
        signal()
        
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    /// signal
    /// signals that the event has occurred
    public func signal() {
        semaphore.signal()
    }
    
    //--------------------------------------------------------------------------
    /// occurred
    /// `true` if the event has occurred
    public private(set) var occurred: Bool {
        get { return occurredAccess.sync { _occurred } }
        set { occurredAccess.sync { _occurred = newValue } }
    }
    
    //--------------------------------------------------------------------------
    /// blockingWait
    /// the first thread goes through the barrier.sync and waits on the
    /// semaphore. When it is signaled `occurred` is set to `true` and all
    /// future threads will exit without waiting
    public func blockingWait(for timeout: TimeInterval) throws {
        try barrier.sync {
            // check if already occurred
            guard !occurred else { return }
            
            // it has not occurred so proceed with the wait
            #if DEBUG
            diagnostic(
                "\(waitString) StreamEvent(\(trackingId)) on \(device.name)",
                categories: .streamSync)
            #endif
            
            if timeout > 0 {
                if semaphore.wait(timeout: .now() + timeout) == .timedOut {
                    #if DEBUG
                    diagnostic("StreamEvent(\(trackingId)) timed out",
                        categories: .streamSync)
                    #endif
                    throw StreamEventError.timedOut
                }
            } else {
                semaphore.wait()
            }
            occurred = true

            #if DEBUG
            diagnostic(
                "\(signaledString) StreamEvent(\(trackingId)) on \(device.name)",
                categories: .streamSync)
            #endif
        }
    }
}
