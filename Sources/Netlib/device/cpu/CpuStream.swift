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
    public var timeout: TimeInterval?
    public var executeAsync: Bool = false
    public var _deviceErrorHandler: DeviceErrorHandler! = nil
    public var _lastDeviceError: DeviceError? = nil
    public var errorMutex: Mutex = Mutex()

    // serial queue
    private let commandQueue: DispatchQueue
    let errorQueue: DispatchQueue
    private let completionEvent: CpuStreamEvent

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo,
                device: ComputeDevice,
                name: String,
                id: Int) throws {
        
        // create serial queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        errorQueue = DispatchQueue(label: "\(name).errorQueue")
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
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        queue {
            throw DeviceError.streamInvalidArgument(idPath: [],
                                                    message: "TestError",
                                                    aux: nil)
        }
    }

    //--------------------------------------------------------------------------
    /// reportDeviceError
    /// sets and propagates a stream error
    func reportDeviceError(_ error: Error) {
        let error = (error as? DeviceError) ??
            .streamError(idPath: [], error: error)
        
        // set the error state
        lastDeviceError = error
        
        // write the error to the log
        logInfo.log.write(level: .error, message: String(describing: error))
        
        // propagate on app thread
        DispatchQueue.main.async {
            self.defaultDeviceErrorHandler(error: error)
        }
        
        // signal the completion event in case the app thread is waiting
        completionEvent.signal()
    }

    //--------------------------------------------------------------------------
    /// tryCatch
    /// tries a throwing function and reports any errors thrown
    func tryCatch<T: DefaultInitializer>(_ body: () throws -> T) -> T {
        guard lastDeviceError == nil else { return T() }
        do {
            return try body()
        } catch {
            reportDeviceError(error)
            return T()
        }
    }
    
    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    ///
    /// This will catch and propagate the last asynchronous error thrown.
    /// I wish there was a better way to do this!
    public func queue(_ body: @escaping () throws -> Void) {
        // if the stream is in an error state, no additional work
        // will be queued
        guard (errorMutex.sync { self._lastDeviceError == nil }) else { return }

        func workFunction() {
            do {
                try body()
            } catch {
                self.reportDeviceError(error)
            }
        }
        
        // queue the work
        if executeAsync {
            commandQueue.async { workFunction() }
        } else {
            workFunction()
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
        commandQueue.async {
            Thread.sleep(forTimeInterval: TimeInterval(seconds))
        }
	}

    //--------------------------------------------------------------------------
    /// blockCallerUntilComplete
    /// blocks the calling thread until the command queue is empty
    public func blockCallerUntilComplete() throws {
        let event = completionEvent
        commandQueue.async {
            event.signal()
        }
        try event.wait(until: timeout)
    }
	
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits for 
	public func wait(for event: StreamEvent) throws {
        let streamEvent = event as! CpuStreamEvent
        try streamEvent.wait(until: timeout)
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
    /// record(event:
	public func record(event: StreamEvent) throws  -> StreamEvent {
        let streamEvent = event as! CpuStreamEvent
        streamEvent.occurred = false

        commandQueue.async {
            streamEvent.signal()
        }
		return event
	}
}
