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
    public var logging: LogInfo?
    public var timeout: TimeInterval?

    // serial queue
    let commandQueue: DispatchQueue
    let completionEvent = CpuStreamEvent(options: StreamEventOptions())

    //--------------------------------------------------------------------------
    // initializers
    public init(logging: LogInfo,
                device: ComputeDevice,
                name: String, id: Int) throws {
        // create serial queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        self.logging = logging
        self.device = device
        self.id = id
        self.name = name
        let path = logging.namePath
        trackingId = ObjectTracker.global.register(self, namePath: path)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
	//--------------------------------------------------------------------------
	/// createEvent
    /// creates an event object used for stream synchronization
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
		return CpuStreamEvent(options: options)
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
        commandQueue.async {
            let streamEvent = event as! CpuStreamEvent
            streamEvent.signal()
        }
		return event
	}
}
