//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public final class CpuComputeService : ComputeService {
	// initializers
	public init(context: EvaluationContext?) throws	{
		self.context = context

		// this is a singleton
		trackingId = ObjectTracker.shared.register(type: self)
		ObjectTracker.shared.markStatic(trackingId: trackingId)

		devices.append(CpuDevice(log: log, service: self, deviceId: 0))
	}
	deinit { ObjectTracker.shared.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public let name = "cpu"
	public var id = 0
	public var devices = [ComputeDevice]()

	// object tracking
	public private(set) var trackingId = 0
	public var namePath: String = "TODO"

	// logging
	public let context: EvaluationContext?
	public var logLevel = LogLevel.error
	public let nestingLevel = 0
}




