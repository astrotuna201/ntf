//******************************************************************************
//  Created by Edward Connell on 11/18/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Cuda

public final class CudaStreamEvent : StreamEvent {
    // properties
    public private(set) var trackingId = 0
    public var recordedTime: Date?

    public let options: StreamEventOptions
    private let timeout: TimeInterval?
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
    public init(options: StreamEventOptions, timeout: TimeInterval?) throws {
        self.options = options
        self.timeout = timeout

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
        _ = cudaEventDestroy(handle)

        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }

    //--------------------------------------------------------------------------
    /// wait
    /// the first thread goes through the barrier.sync and waits on the
    /// semaphore. When it is signaled `occurred` is set to `true` and all
    /// future threads will pass through without waiting
    public func wait() throws {
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
