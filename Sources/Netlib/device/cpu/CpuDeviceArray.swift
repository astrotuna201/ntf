//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public class CpuDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public var buffer: UnsafeMutableRawBufferPointer
    public var device: ComputeDevice
    public var logInfo: LogInfo
    public var version = 0
    private let isReadOnly: Bool
    
    //--------------------------------------------------------------------------
	// initializers
	public init(logInfo: LogInfo, device: ComputeDevice, count: Int) {
        self.device = device
        buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: count, alignment: MemoryLayout<Double>.alignment)
        self.logInfo = logInfo
        self.isReadOnly = false
        self.trackingId = ObjectTracker.global.register(self)
	}

    /// read only uma buffer
    public init(logInfo: LogInfo, device: ComputeDevice,
                buffer: UnsafeRawBufferPointer) {
        assert(buffer.baseAddress != nil)
        self.isReadOnly = true
        self.device = device
        self.logInfo = logInfo
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.trackingId = ObjectTracker.global.register(self)
    }

    /// read write uma buffer
    public init(logInfo: LogInfo, device: ComputeDevice,
                buffer: UnsafeMutableRawBufferPointer) {
        self.isReadOnly = false
        self.device = device
        self.logInfo = logInfo
        self.buffer = buffer
        self.trackingId = ObjectTracker.global.register(self)
    }

    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream) throws {
        assert(!isReadOnly, "cannot mutate read only reference buffer")
        let stream = stream as! CpuStream
        stream.queue {
            self.buffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
	}

	// copyAsync(from deviceArray
	public func copyAsync(from other: DeviceArray,
                          using stream: DeviceStream) throws {
        assert(!isReadOnly, "cannot mutate read only reference buffer")
        assert(buffer.count == other.buffer.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            self.buffer.copyMemory(from: UnsafeRawBufferPointer(other.buffer))
        }
	}

	// copyAsync(from buffer
	public func copyAsync(from buffer: UnsafeRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(!isReadOnly, "cannot mutate read only reference buffer")
        assert(buffer.baseAddress != nil)
        assert(self.buffer.count == buffer.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            self.buffer.copyMemory(from: buffer)
        }
	}

    /// copy(to buffer:
    /// copies the contents of the device array to the host buffer
    /// synchronously with the app thread
	public func copy(to buffer: UnsafeMutableRawBufferPointer,
                     using stream: DeviceStream) throws {
        assert(buffer.baseAddress != nil)
        assert(self.buffer.count == buffer.count, "buffer sizes don't match")
        // wait until the stream is complete then copy
        try stream.waitUntilStreamIsComplete()
        buffer.copyMemory(from: UnsafeRawBufferPointer(self.buffer))
	}

	// copyAsync(to buffer
    /// copies the contents of the device array to the host buffer
    /// asynchronously. It does not wait for the app thread.
	public func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(buffer.baseAddress != nil)
        assert(self.buffer.count == buffer.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            buffer.copyMemory(from: UnsafeRawBufferPointer(self.buffer))
        }
	}
}
