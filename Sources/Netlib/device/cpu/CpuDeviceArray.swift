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
    public var logInfo: LogInfo
    public var version = 0
    
    //--------------------------------------------------------------------------
	// initializers
	public init(logInfo: LogInfo, device: ComputeDevice, count: Int) {
        self.count = count
        self.device = device
        self.data = UnsafeMutableRawPointer.allocate(
            byteCount: count, alignment: MemoryLayout<Double>.alignment)
        self.logInfo = logInfo
        self.trackingId = ObjectTracker.global.register(self)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream) throws {
        let stream = stream as! CpuStream
        stream.queue {
            self.data.initializeMemory(as: UInt8.self,
                                       repeating: 0,
                                       count: self.count)
        }
	}

	// copyAsync(from deviceArray
	public func copyAsync(from other: DeviceArray,
                          using stream: DeviceStream) throws {
        assert(count == other.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            self.data.copyMemory(from: other.data, byteCount: other.count)
        }
	}

	// copyAsync(from buffer
	public func copyAsync(from buffer: UnsafeRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(buffer.baseAddress != nil)
        assert(count == buffer.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            self.data.copyMemory(from: buffer.baseAddress!,
                                 byteCount: buffer.count)
        }
	}

	// copy(to buffer
	public func copy(to buffer: UnsafeMutableRawBufferPointer,
                     using stream: DeviceStream) throws {
        assert(buffer.baseAddress != nil)
        assert(count == buffer.count, "buffer sizes don't match")
        let dataBuffer = UnsafeRawBufferPointer(start: self.data,
                                                count: self.count)
        buffer.copyMemory(from: dataBuffer)
	}

	// copyAsync(to buffer
	public func copyAsync(to buffer: UnsafeMutableRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(buffer.baseAddress != nil)
        assert(count == buffer.count, "buffer sizes don't match")
        let stream = stream as! CpuStream
        stream.queue {
            let dataBuffer = UnsafeRawBufferPointer(start: self.data,
                                                    count: self.count)
            buffer.copyMemory(from: dataBuffer)
        }
	}
}
