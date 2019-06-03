//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public class CudaDeviceArray : DeviceArray {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let buffer: UnsafeMutableRawBufferPointer
    public var device: ComputeDevice { return cudaDevice }
    public var version = 0
    public let isReadOnly: Bool
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
}
