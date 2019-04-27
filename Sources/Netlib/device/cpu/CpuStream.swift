//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream: LocalDeviceStream, StreamGradients {
	// protocol properties
	public private(set) var trackingId = 0
    public let completionEvent: StreamEvent
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var _deviceErrorHandler: DeviceErrorHandler! = nil
    public var _lastError: Error? = nil
    public var errorMutex: Mutex = Mutex()

    // serial queue
    private let commandQueue: DispatchQueue

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo,
                device: ComputeDevice,
                name: String,
                id: Int) {
        
        // create serial queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        completionEvent = CpuStreamEvent(logInfo: logInfo,
                                         options: StreamEventOptions())
        self.logInfo = logInfo
        self.device = device
        self.id = id
        self.name = name
        let path = logInfo.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
        
        // pointer to instance error handler function
        _deviceErrorHandler = defaultDeviceErrorHandler(error:)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    ///
    /// This will catch and propagate the last asynchronous error thrown.
    /// I wish there was a better way to do this!
    public func queue(_ body: @escaping () throws -> Void) {
        // if the stream is in an error state, no additional work
        // will be queued
        guard (errorMutex.sync { self._lastError == nil }) else { return }

        func performBody() {
            do {
                try body()
            } catch {
                self.reportDevice(error: error, event: completionEvent)
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
        return CpuStreamEvent(logInfo: logInfo, options: options)
	}

    /// introduces a delay into command queue processing to simulate workloads
    /// to aid in debugging
	public func debugDelay(seconds: Double) throws {
        queue {
            Thread.sleep(forTimeInterval: TimeInterval(seconds))
        }
	}

    //--------------------------------------------------------------------------
    /// blockCallerUntilComplete
    /// blocks the calling thread until the command queue is empty
    public func blockCallerUntilComplete() throws {
        try record(event: completionEvent).wait(until: timeout)
    }
	
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event is signaled
	public func wait(for event: StreamEvent) throws {
        try event.wait(until: timeout)
	}

    //--------------------------------------------------------------------------
	/// sync(with other
	public func sync(with otherStream: DeviceStream, event: StreamEvent) throws{
        // wait on event to make sure it is clear
        try wait(for: event)
        // record on other stream and wait on event to make sure it is clear
        try wait(for: otherStream.record(event: event))
	}

    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        queue {
            throw DeviceError.streamError(idPath: [], message: "testError")
        }
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
	public func record(event: StreamEvent) throws -> StreamEvent {
        event.occurred = false
        queue {
            event.signal()
        }
        return event
	}
}
