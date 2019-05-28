//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public class CudaDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public var buffer: UnsafeMutableRawBufferPointer
    public var device: ComputeDevice { return cudaDevice }
    public var version = 0
    private let isReadOnly: Bool
    private let cudaDevice: CudaDevice

    //--------------------------------------------------------------------------
    // initializers
	public init(device: CudaDevice, count: Int) throws {
        cudaDevice = device
        self.isReadOnly = false
        try cudaDevice.select()
		var pointer: UnsafeMutableRawPointer?
		try cudaCheck(status: cudaMalloc(&pointer, count))
        buffer = UnsafeMutableRawBufferPointer(start: pointer, count: count)
		trackingId = ObjectTracker.global.register(
                self, supplementalInfo: "count: \(count)")
	}

    //--------------------------------------------------------------------------
    /// readOnly uma buffer
    public init(device: CudaDevice, buffer: UnsafeRawBufferPointer) {
        assert(buffer.baseAddress != nil)
        cudaDevice = device
        self.isReadOnly = true
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    /// readWrite uma buffer
    public init(device: CudaDevice, buffer: UnsafeMutableRawBufferPointer) {
        cudaDevice = device
        self.isReadOnly = false
        self.buffer = buffer
        self.trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    // cleanup
    deinit {
		do {
			try cudaDevice.select()
			try cudaCheck(status: cudaFree(buffer.baseAddress!))
			ObjectTracker.global.remove(trackingId: trackingId)
		} catch {
            cudaDevice.writeLog(
                    "\(releaseString) CudaDeviceArray(\(trackingId)) " +
                    "\(String(describing: error))")
		}
	}

	//--------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream) throws {
        let cudaStream = (stream as! CudaStream).handle
        try cudaCheck(status: cudaMemsetAsync(buffer.baseAddress!, Int32(0),
                                              buffer.count, cudaStream))
	}

	//--------------------------------------------------------------------------
	// copyAsync(from deviceArray
    public func copyAsync(from other: DeviceArray,
                          using stream: DeviceStream) throws {
        assert(!isReadOnly, "cannot mutate read only reference buffer")
        assert(buffer.count == other.buffer.count, "buffer sizes don't match")
		assert(other is CudaDeviceArray)
		let stream = stream as! CudaStream
		try stream.cudaDevice.select()

		// copy
        try cudaCheck(status: cudaMemcpyAsync(
                buffer.baseAddress!,
                UnsafeRawPointer(other.buffer.baseAddress!),
                buffer.count,
                cudaMemcpyDeviceToDevice, stream.handle))
    }
	
	//--------------------------------------------------------------------------
	// copyAsync(from buffer
    public func copyAsync(from otherBuffer: UnsafeRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(!isReadOnly, "cannot mutate read only reference buffer")
        assert(otherBuffer.baseAddress != nil)
        assert(self.buffer.count == otherBuffer.count, "buffer sizes don't match")
		let stream = stream as! CudaStream
		try stream.cudaDevice.select()

		try cudaCheck(status: cudaMemcpyAsync(
                buffer.baseAddress!,
                UnsafeRawPointer(otherBuffer.baseAddress!),
			    buffer.count,
                cudaMemcpyHostToDevice, stream.handle))
	}
	
	//--------------------------------------------------------------------------
	/// copyAsync(to buffer
    public func copyAsync(to otherBuffer: UnsafeMutableRawBufferPointer,
                          using stream: DeviceStream) throws {
        assert(otherBuffer.baseAddress != nil)
        assert(otherBuffer.count == buffer.count, "buffer sizes don't match")
		let stream = stream as! CudaStream
		try stream.cudaDevice.select()

        try cudaCheck(status: cudaMemcpyAsync(
                otherBuffer.baseAddress!,
                UnsafeRawPointer(buffer.baseAddress!),
                buffer.count,
                cudaMemcpyDeviceToHost, stream.handle))
	}
}
