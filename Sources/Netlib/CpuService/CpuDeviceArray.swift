//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public class CpuDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public var count: Int
    public var data: UnsafeMutableRawPointer
    public var device: ComputeDevice
    public var logging: LogInfo?
    public var version = 0
    
    //--------------------------------------------------------------------------
	// initializers
	public init(logging: LogInfo, device: ComputeDevice, count: Int) {
        self.count = count
        self.data = UnsafeMutableRawPointer(bitPattern: 0)!
		self.device = device
        self.logging = logging
		self.trackingId = ObjectTracker.global.register(self)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream?) throws {

	}

	// copyAsync(from deviceArray
	public func copyAsync(from other: DeviceArray,
                          using stream: DeviceStream) throws {

	}

	// copyAsync(from buffer
	public func copyAsync(from buffer: UnsafeRawBufferPointer,
                          using stream: DeviceStream) throws {

	}

	// copy(to buffer
	public func copy(to buffer: UnsafeMutableRawBufferPointer,
                     using stream: DeviceStream) throws {

	}

	// copyAsync(to buffer
	public func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                          using stream: DeviceStream) throws {

	}
}
