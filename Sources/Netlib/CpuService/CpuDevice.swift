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
    public var logging: LogInfo?
    public var maxThreadsPerBlock: Int { return 1 }
    public let name: String = "cpu"
    public weak var service: ComputeService!
    private let streamId = AtomicCounter()
    public let usesUnifiedAddressing = true
    public var utilization: Float = 0

    // TODO this should be currently available and not physicalMemory
    public lazy var availableMemory: UInt64 = {
        return ProcessInfo().physicalMemory
    }()

    //--------------------------------------------------------------------------
	// initializers
	public init(logging: LogInfo, service: CpuComputeService, deviceId: Int) {
		self.logging = logging
		self.id = deviceId
		self.service = service

		// devices are statically held by the Platform.service
        trackingId = ObjectTracker.global.register(self,
                                                   namePath: logging.namePath,
                                                   isStatic: true)
	}
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//-------------------------------------
	// createArray
	//	This creates memory on the device
	public func createArray(count: Int) throws -> DeviceArray {
		return CpuDeviceArray(logging: logging!, device: self, count: count)
	}

	//-------------------------------------
	// createStream
	public func createStream(name streamName: String) throws -> DeviceStream {
        let id = streamId.increment()
        return try CpuStream(logging: logging!.child("\(streamName):\(id)"),
                             device: self, name: streamName, id: id)
	}
} // CpuDevice


