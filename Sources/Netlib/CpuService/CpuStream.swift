//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public final class CpuStream : DeviceStream {
	//--------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logging: LogInfo?

    //--------------------------------------------------------------------------
    // initializers
    public init(logging: LogInfo,
                device: ComputeDevice,
                name: String, id: Int) throws {
        // init
        self.logging = logging
        self.device = device
        self.id = id
        self.name = name
        trackingId = ObjectTracker.global.register(self,
                                                   namePath: logging.namePath)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func execute<T>(functionId: UUID, with parameters: T) throws {
        
    }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func setup<T>(functionId: UUID, instanceId: UUID,
                         with parameters: T) throws {
        
    }
    
    //--------------------------------------------------------------------------
    // createEvent
    public func release(instanceId: UUID) throws {
        
    }

	//--------------------------------------------------------------------------
	// createEvent
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
		return CpuStreamEvent(options: options)
	}

	public func debugDelay(seconds: Double) throws {
        
	}

	// blockCallerUntilComplete
	public func blockCallerUntilComplete() throws {
		
	}
	
	public func wait(for event: StreamEvent) throws {
		
	}

	// sync(with other
	public func sync(with other: DeviceStream, event: StreamEvent) throws {
	}

	public func record(event: StreamEvent) throws  -> StreamEvent {
		return event
	}

}

//==============================================================================
// CpuStreamEvent
final public class CpuStreamEvent : StreamEvent {
	public required init(options: StreamEventOptions) {
		trackingId = ObjectTracker.global.register(self, namePath: logging?.namePath)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// properties
	public private (set) var trackingId = 0
	public var occurred: Bool { return true }
    public var logging: LogInfo? = nil
}
