//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream: LocalDeviceStream, StreamGradients {
	// protocol properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval = 0
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
    public init(logInfo: LogInfo,
                device: ComputeDevice,
                name: String,
                id: Int) {
        
        // create serial command queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        
        // create a completion event
        self.logInfo = logInfo
        self.device = device
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        let path = logInfo.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    /// TODO: remove redundancy with 2 param version below!
    ///
    public func queue<T1, R>(_ functionName: @autoclosure () -> String,
                            _ t1: T1, _ result: inout R,
                            _ body: @escaping
        (TensorViewCollection<T1>, inout TensorViewMutableCollection<R>) throws
        -> Void) where T1: TensorView, R: TensorView
    {
        // if the stream is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }

        // schedule the work
        diagnostic("\(schedulingString): \(functionName())",
            categories: .scheduling)
        let currentNestingLevel = logInfo.nestingLevel
        logInfo.nestingLevel = currentNestingLevel + 1
        
        do {
            // get the parameter sequences
            let t1 = try t1.values(using: self)
            var ref = try result.reference(using: self)
            var results = try ref.mutableValues(using: self)

            if executeSynchronously {
                try body(t1, &results)
            } else {
                // safely set a new completion event on the result
                let completionEvent =
                    try ref.createAndSetCompletionEvent(using: self)
                
                // queue the work
                commandQueue.async {
                    do {
                        try body(t1, &results)
                    } catch {
                        self.reportDevice(error: error)
                    }
                }
                diagnostic(">>> \(functionName()) is queued",
                           categories: .scheduling)
                
                // queue signaling of the completion event after the work
                // is complete
                try record(event: completionEvent)
            }
        } catch {
            self.reportDevice(error: error)
        }
        
        // put back
        logInfo.nestingLevel = currentNestingLevel
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    public func queue<T1, T2, R>(_ functionName: @autoclosure () -> String,
                                 _ t1: T1, _ t2: T2, _ result: inout R,
                                 _ body: @escaping
        (TensorViewCollection<T1>, TensorViewCollection<T2>,
        inout TensorViewMutableCollection<R>) throws -> Void)
        where T1: TensorView, T2: TensorView, R: TensorView
    {
        // if the stream is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        
        // schedule the work
        diagnostic("\(schedulingString): \(functionName())",
            categories: .scheduling)
        let currentNestingLevel = logInfo.nestingLevel
        logInfo.nestingLevel = currentNestingLevel + 1
        
        do {
            // get the parameter sequences
            let t1 = try t1.values(using: self)
            let t2 = try t2.values(using: self)
            var ref = try result.reference(using: self)
            var results = try ref.mutableValues(using: self)
            
            if executeSynchronously {
                try body(t1, t2, &results)
            } else {
                // safely set a new completion event on the result
                let completionEvent =
                    try ref.createAndSetCompletionEvent(using: self)
                
                // queue the work
                commandQueue.async {
                    do {
                        try body(t1, t2, &results)
                    } catch {
                        self.reportDevice(error: error)
                    }
                }
                diagnostic(">>> \(functionName()) is queued",
                    categories: .scheduling)
                
                // queue signaling of the completion event after the work
                // is complete
                try record(event: completionEvent)
            }
        } catch {
            self.reportDevice(error: error)
        }
        
        // put back
        logInfo.nestingLevel = currentNestingLevel
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    public func queue(_ body: @escaping () throws -> Void) {
        // if the stream is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        
        func performBody() {
            do {
                try body()
            } catch {
                self.reportDevice(error: error)
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
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        return CpuStreamEvent(stream: self, options: options)
	}

    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    public func record(event: StreamEvent) throws -> StreamEvent {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        event.occurred = false
        queue {
            event.signal()
        }
        return event
    }

    //--------------------------------------------------------------------------
    /// futureWait(for event:
    /// waits until the event is signaled
	public func futureWait(for event: StreamEvent) throws {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        let timeout = self.timeout
        queue {
            try event.blockingWait(for: timeout)
        }
	}

    //--------------------------------------------------------------------------
    /// waitUntilStreamIsComplete
    /// blocks the calling thread until the command queue is empty
    public func waitUntilStreamIsComplete() throws {
        assert(Thread.current === creatorThread, streamThreadViolationMessage)
        try record(event: createEvent()).blockingWait(for: timeout)
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
