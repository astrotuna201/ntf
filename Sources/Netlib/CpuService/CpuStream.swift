//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream : DeviceStream {
    //--------------------------------------------------------------------------
	// protocol properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logging: LogInfo
    public var timeout: TimeInterval?
    /// the last error thrown by a stream function
    public private(set) var lastStreamError: Error?
    public var executeAsync: Bool = false

    // serial queue
    let commandQueue: DispatchQueue
    let errorQueue: DispatchQueue
    let completionEvent: CpuStreamEvent

    //--------------------------------------------------------------------------
    // initializers
    public init(logging: LogInfo,
                device: ComputeDevice,
                name: String, id: Int) throws {
        // create serial queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        errorQueue = DispatchQueue(label: "\(name).errorQueue")
        completionEvent = CpuStreamEvent(logging: logging,
                                         options: StreamEventOptions())
        self.logging = logging
        self.device = device
        self.id = id
        self.name = name
        let path = logging.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
    /// queues a closure on the stream for execution
    ///
    /// This will catch and propagate the last asynchronous error thrown.
    /// I wish there was a better way to do this!
    public func queue(_ body: @escaping () throws -> Void) throws {
        // check for a pending error from the last operation
        try errorQueue.sync {
            guard lastStreamError == nil else { throw lastStreamError! }
        }
        
        // queue the work
        if executeAsync {
            commandQueue.async {
                do {
                    try body()
                } catch {
                    self.errorQueue.sync {
                        self.lastStreamError = error
                        self.writeLog(String(describing: error))
                    }
                }
            }
        } else {
            try body()
        }
    }

	//--------------------------------------------------------------------------
	/// createEvent
    /// creates an event object used for stream synchronization
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
        return CpuStreamEvent(logging: logging, options: options)
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
