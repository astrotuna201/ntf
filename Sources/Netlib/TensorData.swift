//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

public typealias BufferUInt8 = UnsafeBufferPointer<UInt8>
public typealias MutableBufferUInt8 = UnsafeMutableBufferPointer<UInt8>

public class TensorData<Scalar> : ObjectTracking, Logging {
    //----------------------------------------------------------------------------
    // properties
    public let accessQueue = DispatchQueue(label: "TensorData.accessQueue")
    public let elementCount: Int
    public let elementSize = MemoryLayout<Scalar>.size
    public var byteCount: Int { return elementSize * elementCount }
    public var autoReleaseUmaBuffer = false

    // object tracking
    public private(set) var trackingId = 0
    public var namePath: String = "TODO"

    // logging
    public let log: Log?
    public var logLevel = LogLevel.error
    public let nestingLevel = 0

    /// an optional context specific name for logging
    private var _name: String?
    public var name: String {
        get { return _name ?? String(describing: TensorData<Scalar>.self) }
        set { _name = newValue }
    }

    // testing
    public private(set) var lastAccessCopiedBuffer = false

    // local
    private let streamRequired = "stream is required for device data transfers"
    private let isReadOnlyReference: Bool
    private var hostArray = [UInt8]()
    private var hostBuffer: MutableBufferUInt8!
    private var hostVersion = -1

    // stream sync
    private var _streamSyncEvent: StreamEvent!
    private func getSyncEvent(using stream: DeviceStream) throws -> StreamEvent {
        if _streamSyncEvent == nil {
            _streamSyncEvent = try stream.createEvent(options: [])
        }
        return _streamSyncEvent
    }

    // this can either point to the hostArray or to the deviceArray
    // depending on the location of the master
    private var deviceDataPointer: UnsafeMutableRawPointer!

    // this is indexed by [service.id][device.id]
    // and contains a lazy allocated array on each device,
    // which is a replica of the current master
    private var deviceArrays = [[ArrayInfo?]]()

    public class ArrayInfo {
        public init(array: DeviceArray, stream: DeviceStream) {
            self.array = array
            self.stream = stream
        }

        public let array: DeviceArray
        // stream is tracked for synchronous cleanup (deinit) of the array
        public var stream: DeviceStream
    }

    // whenever a buffer write pointer is taken, the associated DeviceArray
    // becomes the master copy for replication. Synchronization across threads
    // is still required for taking multiple write pointers, however
    // this does automatically synchronize data migrations.
    // A value of nil means that the master is the umaBuffer
    public private(set) var master: ArrayInfo?

    // this is incremented each time a write pointer is taken
    // all replicated buffers will stay in sync with this version
    private var masterVersion = -1

    //--------------------------------------------------------------------------
    // initializers
    public convenience init() {
        self.init(log: nil, elementCount: 0)
    }

    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a database
    public init(log: Log?, readOnlyReferenceTo buffer: BufferUInt8) {
        self.log  = log
        isReadOnlyReference = true
        elementCount  = buffer.count
        masterVersion = 0
        hostVersion   = 0

        // we won't ever actually mutate in this case
        hostBuffer = MutableBufferUInt8(
                start: UnsafeMutablePointer(mutating: buffer.baseAddress),
                count: buffer.count)

        register()
    }

    //----------------------------------------
    // create new space
    public init(log: Log?, elementCount: Int, name: String? = nil) {
        isReadOnlyReference = false
        self.log  = log
        self.elementCount = elementCount
        self._name = name
        register()
    }

    // object lifetime tracking for leak detection
    private func register() {
        trackingId = objectTracker.register(
                type: self, info: "elementCount: \(elementCount)")

        if elementCount > 0 && willLog(level: .diagnostic) {
            diagnostic("\(createString) \(name)(\(trackingId)) " +
                    "elements: \(elementCount)", categories: .dataAlloc)
        }
    }
}
