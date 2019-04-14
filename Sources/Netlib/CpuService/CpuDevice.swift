//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

public class CpuDevice : ComputeDevice {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public let attributes = [String : String]()
    public let id: Int
    public var logInfo: LogInfo
    public var maxThreadsPerBlock: Int { return 1 }
    public let name: String
    public weak var service: ComputeService!
    private let streamId = AtomicCounter(value: -1)
    public var timeout: TimeInterval?
    public let memoryAddressing: MemoryAddressing
    public var utilization: Float = 0

    // TODO this should be currently available and not physicalMemory
    public lazy var availableMemory: UInt64 = {
        return ProcessInfo().physicalMemory
    }()

    //--------------------------------------------------------------------------
	// initializers
	public init(service: CpuComputeService,
                deviceId: Int,
                logInfo: LogInfo,
                memoryAddressing: MemoryAddressing,
                timeout: TimeInterval? = nil) {
        self.name = "cpu:\(deviceId)"
		self.logInfo = logInfo
		self.id = deviceId
		self.service = service
        self.timeout = timeout
        self.memoryAddressing = memoryAddressing

		// devices are statically held by the Platform.service
        trackingId = ObjectTracker.global.register(self,
                                                   namePath: logNamePath,
                                                   isStatic: true)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//-------------------------------------
	// createArray
	//	This creates memory on the device
	public func createArray(count: Int) throws -> DeviceArray {
        return CpuDeviceArray(logInfo: logInfo, device: self, count: count)
	}

	//-------------------------------------
	// createStream
	public func createStream(name streamName: String) throws -> DeviceStream {
        let id = streamId.increment()
        let streamName = "\(streamName):\(id)"
        return try CpuStream(logInfo: logInfo.child(streamName),
                             device: self, name: streamName, id: id)
	}
} // CpuDevice


