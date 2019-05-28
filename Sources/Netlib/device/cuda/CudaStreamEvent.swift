//******************************************************************************
//  Created by Edward Connell on 11/18/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Cuda

public final class CudaStreamEvent : StreamEvent {
    // properties
    public private(set) var trackingId = 0
    public private(set) var recordedTime: Date?

    public let options: StreamEventOptions
    public let handle: cudaEvent_t
    private let barrier = Mutex()
    private let semaphore = DispatchSemaphore(value: 0)

    //--------------------------------------------------------------------------
    // occurred
    public var occurred: Bool {
        return cudaEventQuery(handle) == cudaSuccess
    }

    //--------------------------------------------------------------------------
    // initializers
    public init(options: StreamEventOptions) throws {
        self.options = options

        // the default is non host blocking, non timing, non inter process
        var flags: Int32 = cudaEventDisableTiming
        if !options.contains(.timing)      { flags &= ~cudaEventDisableTiming }
        if options.contains(.interprocess) { flags |= cudaEventInterprocess |
                cudaEventDisableTiming }

        var temp: cudaEvent_t?
        try cudaCheck(status: cudaEventCreateWithFlags(&temp, UInt32(flags)))
        handle = temp!

        #if DEBUG
        trackingId = ObjectTracker.global.register(self)
        #endif
    }

    //--------------------------------------------------------------------------
    // deinit
    deinit {
        // signal if anyone was waiting
        signal()
        _ = cudaEventDestroy(handle)

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
//        try barrier.sync {
//            guard !occurred else { return }
//            if timeout > 0 {
//                if semaphore.wait(timeout: .now() + timeout) == .timedOut {
//                    throw StreamEventError.timedOut
//                }
//            } else {
//                semaphore.wait()
//            }
//        }
    }
}
