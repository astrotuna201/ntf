//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public class CpuDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let buffer: UnsafeMutableRawBufferPointer
    public let device: ComputeDevice
    public var version = 0
    public let isReadOnly: Bool
    
    //--------------------------------------------------------------------------
	/// with count
	public init(device: ComputeDevice, count: Int) {
        self.device = device
        buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: count, alignment: MemoryLayout<Double>.alignment)
        self.isReadOnly = false
        self.trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    /// readOnly uma buffer
    public init(device: ComputeDevice, buffer: UnsafeRawBufferPointer) {
        assert(buffer.baseAddress != nil)
        self.isReadOnly = true
        self.device = device
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    /// readWrite uma buffer
    public init(device: ComputeDevice, buffer: UnsafeMutableRawBufferPointer) {
        self.isReadOnly = false
        self.device = device
        self.buffer = buffer
        self.trackingId = ObjectTracker.global.register(self)
    }

    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
