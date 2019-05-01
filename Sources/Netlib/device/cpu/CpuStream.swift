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

    // serial queue
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
        let path = logInfo.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    /// This will catch and propagate the last asynchronous error thrown.
    public func queue<T>(_ result: inout T,
                         _ body: @escaping (inout T) throws -> Void)
        where T: TensorView
    {
        // if the stream is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        do {
            var ref = try result.reference(using: self)
            if executeSynchronously {
                try body(&ref)
            } else {
                // safely set a new completion event on the result
                let completionEvent =
                    try ref.createAndSetCompletionEvent(using: self)
                
                // queue the work
                commandQueue.async {
                    do {
                        try body(&ref)
                    } catch {
                        self.reportDevice(error: error)
                    }
                }
                diagnostic(">>> function queued",
                           categories: .scheduling, indent: 1)
                
                // queue signaling of the completion event after the work
                // is complete
                try record(event: completionEvent)
            }
        } catch {
            self.reportDevice(error: error)
        }
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
        return CpuStreamEvent(stream: self, options: options)
	}

    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    public func record(event: StreamEvent) throws -> StreamEvent {
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
        let timeout = self.timeout
        queue {
            try event.blockingWait(for: timeout)
        }
	}

    //--------------------------------------------------------------------------
    /// waitUntilStreamIsComplete
    /// blocks the calling thread until the command queue is empty
    public func waitUntilStreamIsComplete() throws {
        try record(event: createEvent()).blockingWait(for: timeout)
    }
    
    //--------------------------------------------------------------------------
    /// introduces a delay into command queue processing to simulate workloads
    /// to aid in debugging
    public func debugDelay(seconds: Double) throws {
        queue {
            Thread.sleep(forTimeInterval: TimeInterval(seconds))
        }
    }
    
    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        queue {
            throw DeviceError.streamError(idPath: [], message: "testError")
        }
    }
}
