//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream: LocalDeviceStream, StreamGradients {
	// protocol properties
	public private(set) var trackingId = 0
    public let accessQueue = DispatchQueue(label: "CpuStream.accessQueue")
    public var defaultStreamEventOptions = StreamEventOptions()
	public let device: ComputeDevice
    public let id = Platform.nextUniqueStreamId
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    
    /// used to detect accidental stream access by other threads
    private let creatorThread: Thread
    /// the queue used for command execution
    private let commandQueue: DispatchQueue

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo, device: ComputeDevice, name: String) {
        // create serial command queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        
        // create a completion event
        self.logInfo = logInfo
        self.device = device
        self.name = name
        self.creatorThread = Thread.current
        let path = logInfo.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
        
        diagnostic("\(createString) DeviceStream(\(trackingId)) " +
            "\(device.name)_\(name)", categories: .streamAlloc)
    }
    
    //--------------------------------------------------------------------------
    /// deinit
    /// waits for the queue to finish
    deinit {
        assert(Thread.current === creatorThread,
               "Stream has been captured and is being released by a " +
            "different thread. Probably by a queued function on the stream.")

        diagnostic("\(releaseString) DeviceStream(\(trackingId)) " +
            "\(device.name)_\(name)", categories: [.streamAlloc])
        
        // release
        ObjectTracker.global.remove(trackingId: trackingId)

        // wait for the command queue to complete before shutting down
        do {
            try waitUntilStreamIsComplete()
        } catch {
            if let timeout = self.timeout {
                diagnostic("\(timeoutString) DeviceStream(\(trackingId)) " +
                        "\(device.name)_\(name) timeout: \(timeout)",
                        categories: [.streamAlloc])
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    ///
    public func queue<Inputs, R>(
        _ functionName: @autoclosure () -> String,
        _ inputs: () throws -> Inputs,
        _ result: inout R,
        _ body: @escaping (Inputs, inout R.MutableValues) throws
        -> Void) where R: TensorView
    {
        return accessQueue.sync {
            // if the stream is in an error state, no additional work
            // will be queued
            guard lastError == nil else { return }
            
            // schedule the work
            diagnostic("\(schedulingString): \(functionName())",
                categories: .scheduling)
            
            do {
                // get the parameter sequences
                let input = try inputs()
                var sharedView = try result.sharedView(using: self)
                var results = try sharedView.mutableValues(using: self)
                
                if executeSynchronously {
                    try body(input, &results)
                } else {
                    // queue the work
                    // report to device so we don't take a reference to `self`
                    let errorDevice = device
                    commandQueue.async {
                        do {
                            try body(input, &results)
                        } catch {
                            errorDevice.reportDevice(error: error)
                        }
                    }
                    diagnostic("\(schedulingString): \(functionName()) complete",
                        categories: .scheduling)
                }
            } catch {
                self.reportDevice(error: error)
            }
        }
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    public func queue(_ body: @escaping () throws -> Void) {
        return accessQueue.sync {
            unsafeQueue(body)
        }
    }

    //--------------------------------------------------------------------------
    private func unsafeQueue(_ body: @escaping () throws -> Void) {
        // if the stream is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        let errorDevice = device
        
        // make sure not to capture `self`
        func performBody() {
            do {
                try body()
            } catch {
                errorDevice.reportDevice(error: error)
            }
        }
        
        // queue the work
        if executeSynchronously {
            performBody()
        } else {
            commandQueue.async { performBody() }
        }
    }
    
    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for stream synchronization
    public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
        return accessQueue.sync {
            let event = CpuStreamEvent(options: options, timeout: timeout)
            diagnostic("\(createString) StreamEvent(\(event.trackingId)) on " +
                "\(device.name)_\(name)", categories: .streamAlloc)
            return event
        }
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    public func record(event: StreamEvent) throws -> StreamEvent {
        let event = event as! CpuStreamEvent
        return try accessQueue.sync {
            guard lastError == nil else { throw lastError! }
            diagnostic("\(recordString) StreamEvent(\(event.trackingId)) on " +
                "\(device.name)_\(name)", categories: .streamSync)
            
            // set event time
            if defaultStreamEventOptions.contains(.timing) {
                event.recordedTime = Date()
            }
            
            unsafeQueue {
                event.signal()
            }
            return event
        }
    }

    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    public func wait(for event: StreamEvent) throws {
        guard !event.occurred else { return }
        return try accessQueue.sync {
            guard lastError == nil else { throw lastError! }
            diagnostic("\(waitString) StreamEvent(\(event.trackingId)) on " +
                "\(device.name)_\(name)", categories: .streamSync)
            
            unsafeQueue {
                try event.wait()
            }
        }
    }

    //--------------------------------------------------------------------------
    /// waitUntilStreamIsComplete
    /// blocks the calling thread until the command queue is empty
    public func waitUntilStreamIsComplete() throws {
        let event = try record(event: createEvent())
        diagnostic("\(waitString) StreamEvent(\(event.trackingId)) " +
            "waiting for \(device.name)_\(name) to complete",
            categories: .streamSync)
        try event.wait()
        diagnostic("\(signaledString) StreamEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .streamSync)
    }
    
    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the stream by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
        where T: TensorView
    {
        let delay = TimeInterval(x.shape.elementCount) * timePerElement
        delayStream(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayStream(atLeast:
    /// causes the stream to sleep for the specified interval for testing
    public func delayStream(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        queue {
            Thread.sleep(forTimeInterval: interval)
        }
    }
    
    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        queue {
            throw DeviceError.streamError(idPath: [], message: "testError")
        }
    }
}
