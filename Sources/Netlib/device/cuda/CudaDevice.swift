//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Cuda

public class CudaDevice : LocalComputeDevice {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public private (set) weak var service: ComputeService!
    public let attributes = [String : String]()
    public var deviceArrayReplicaKey: DeviceArrayReplicaKey
    public let id: Int
    public var logInfo: LogInfo
    public var maxThreadsPerBlock: Int { return Int(props.maxThreadsPerBlock) }
    public let name: String
    private let streamId = AtomicCounter(value: -1)
    public var timeout: TimeInterval?
    public let memoryAddressing: MemoryAddressing
    public var utilization: Float = 0
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()

    // TODO this should be currently available and not physicalMemory
    public lazy var availableMemory: UInt64 = {
        return ProcessInfo.processInfo.physicalMemory
    }()

    private let props: cudaDeviceProp

    //--------------------------------------------------------------------------
	// initializers
    public init(service: CudaComputeService,
                deviceId: Int,
                logInfo: LogInfo,
                memoryAddressing: MemoryAddressing,
                timeout: TimeInterval?) throws {
        self.name = "cpu:\(deviceId)"
        self.logInfo = logInfo
        self.id = deviceId
        self.service = service
        self.timeout = timeout
        self.memoryAddressing = memoryAddressing
        deviceArrayReplicaKey =
                DeviceArrayReplicaKey(platformId: service.platform!.id,
                                      serviceId: service.id, deviceId: id)

        // get device properties
        var tempProps = cudaDeviceProp()
        try cudaCheck(status: cudaGetDeviceProperties(&tempProps, 0))
        props = tempProps
        let nameCapacity = MemoryLayout.size(ofValue: tempProps.name)

        name = withUnsafePointer(to: &tempProps.name) {
            $0.withMemoryRebound(to: UInt8.self, capacity: nameCapacity) {
                String(cString: $0)
            }
        }

        // initialize attribute list
        attributes = [
            "name"               : self.name,
            "compute capability" : "\(props.major).\(props.minor)",
            "global memory"      : "\(props.totalGlobalMem / (1024 * 1024)) MB",
            "multiprocessors"    : "\(props.multiProcessorCount)",
            "unified addressing" : "\(memoryAddressing == .unified ? "yes":"no")"
        ]

        // devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
                .register(self, namePath: logNamePath, isStatic: true)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// createArray
	public func createArray(count: Int) throws -> DeviceArray {
		return try CudaDeviceArray(device: self, count: count)
	}

    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a device array from a uma buffer.
    public func createMutableReferenceArray(
            buffer: UnsafeMutableRawBufferPointer) -> DeviceArray {
        return CudaDeviceArray(device: self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a device array from a uma buffer.
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
                    -> DeviceArray {
        return CudaDeviceArray(device: self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
	// createStream
    public func createStream(name streamName: String) throws -> DeviceStream {
        let id = streamId.increment()
        let streamName = "\(streamName):\(id)"
        return try CudaStream(logInfo: logInfo.flat(streamName),
                              device: self, name: streamName, id: id)
    }

	//--------------------------------------------------------------------------
	// select
	public func select() throws {
		try cudaCheck(status: cudaSetDevice(Int32(id)))
	}
}
