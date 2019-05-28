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
        var temp: cudaEvent_t?
        try cudaCheck(status: cudaEventCreateWithFlags(
                &temp, UInt32(cudaEventDisableTiming)))
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
    public func wait() throws {
        try barrier.sync {
            guard !occurred else { return }
            if let timeout = timeout {
                var elapsed = 0.0;
                // TODO: is 1 millisecond reasonable?
                let pollingInterval = 0.001
                while (cudaEventQuery(handle) != cudaSuccess) {
                    Thread.sleep(forTimeInterval: pollingInterval)
                    elapsed += pollingInterval;
                    if (elapsed >= timeout) {
                        throw StreamEventError.timedOut
                    }
                }
            } else {
                // wait indefinitely
                try cudaCheck(status: cudaEventSynchronize(handle))
            }
        }
    }
}
