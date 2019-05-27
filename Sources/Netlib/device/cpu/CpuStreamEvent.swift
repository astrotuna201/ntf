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
    public private(set) var trackingId = 0
    public private(set) var occurred: Bool = false
    public private(set) var recordedTime: Date?

    public let options: StreamEventOptions
    private let barrier = Mutex()
    private let semaphore = DispatchSemaphore(value: 0)

    //--------------------------------------------------------------------------
    // initializers
    public init(options: StreamEventOptions) {
        self.options = options
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)
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
    /// measure elapsed time since another event
    public func elapsedTime(since event: StreamEvent) -> TimeInterval {
        guard let eventTime = event.recordedTime else { return 0 }
        return Date().timeIntervalSince(eventTime)
    }

    //--------------------------------------------------------------------------
    /// tells the event it is being recorded
    public func record() {
        recordedTime = Date()
    }

    //--------------------------------------------------------------------------
    /// signal
    /// signals that the event has occurred
    public func signal() {
        semaphore.signal()
    }
    
    //--------------------------------------------------------------------------
    /// wait
    /// the first thread goes through the barrier.sync and waits on the
    /// semaphore. When it is signaled `occurred` is set to `true` and all
    /// future threads will pass through without waiting
    public func wait(until timeout: TimeInterval) throws {
        try barrier.sync {
            guard !occurred else { return }
            if timeout > 0 {
                if semaphore.wait(timeout: .now() + timeout) == .timedOut {
                    throw StreamEventError.timedOut
                }
            } else {
                semaphore.wait()
            }
            occurred = true
        }
    }
}
