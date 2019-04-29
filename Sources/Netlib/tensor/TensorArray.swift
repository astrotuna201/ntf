//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorArray
/// The TensorArray object is a flat array of scalars used by the TensorView.
/// It is responsible for replication and syncing between devices.
/// It is not created or directly used by end users.
final public class TensorArray: ObjectTracking, Logging {
    //--------------------------------------------------------------------------
    /// used by TensorViews to synchronize access to this object
    public let accessQueue = DispatchQueue(label: "TensorArray.accessQueue")
    /// the size of the data array in bytes
    public let byteCount: Int
    /// `true` if the data array references an existing read only buffer
    public let isReadOnly: Bool
    /// testing: `true` if the last access caused the contents of the
    /// buffer to be copied
    public private(set) var lastAccessCopiedBuffer = false
    /// testing: is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied. It's stored here instead of on the
    /// view, because the view is immutable when taking a read only pointer
    public var lastAccessMutatedView: Bool = false
    /// the last stream that had readWrite access
    public var lastReadWriteStream: DeviceStream?
    /// whenever a buffer write pointer is taken, the associated DeviceArray
    /// becomes the master copy for replication. Synchronization across threads
    /// is still required for taking multiple write pointers, however
    /// this does automatically synchronize data migrations.
    /// The value will be `nil` if no access has been taken yet
    private var master: DeviceArray?
    /// this is incremented each time a write pointer is taken
    /// all replicated buffers will stay in sync with this version
    private var masterVersion = -1
    /// name label used for logging
    public let name: String
    /// replication collection
    private var replicas = [DeviceArrayReplicaKey : DeviceArray]()
    /// the object tracking id
    public private(set) var trackingId = 0

    //--------------------------------------------------------------------------
    // initializers
    public init() {
        byteCount = 0
        isReadOnly = false
        name = ""
    }
    
    //----------------------------------------
    // create new array based on scalar size
    public init<Scalar>(type: Scalar.Type, count: Int, name: String) {
        self.name = name
        isReadOnly = false
        byteCount = count * MemoryLayout<Scalar>.size
        register(type: type, count: count)
        
        diagnostic("\(createString) \(name)(\(trackingId)) " +
            "\(String(describing: Scalar.self))[\(count)]",
            categories: .dataAlloc)
    }

    //----------------------------------------
    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a read only data object
    public init<Scalar>(referenceTo buffer: UnsafeBufferPointer<Scalar>,
                        name: String) {
        self.name = name
        masterVersion = 0
        isReadOnly = true
        byteCount = buffer.count
        let stream = _Streams.local.appThreadStream

        // create the replica device array
        let key = stream.device.deviceArrayReplicaKey
        let bytes = UnsafeRawBufferPointer(buffer)
        let array = stream.device.createReferenceArray(buffer: bytes)
        array.version = -1
        replicas[key] = array
        register(type: Scalar.self, count: buffer.count)

        diagnostic("\(referenceString) \(name)(\(trackingId)) " +
            "readOnly device array reference on \(stream.device.name) " +
            "\(String(describing: Scalar.self))[\(buffer.count)]",
            categories: .dataAlloc)
    }
    
    //----------------------------------------
    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a read only data object
    public init<Scalar>(referenceTo buffer: UnsafeMutableBufferPointer<Scalar>,
                        name: String) {
        self.name = name
        masterVersion = 0
        isReadOnly = false
        byteCount = buffer.count
        let stream = _Streams.local.appThreadStream

        // create the replica device array
        let key = stream.device.deviceArrayReplicaKey
        let bytes = UnsafeMutableRawBufferPointer(buffer)
        let array = stream.device.createMutableReferenceArray(buffer: bytes)
        array.version = -1
        replicas[key] = array
        register(type: Scalar.self, count: buffer.count)

        diagnostic("\(referenceString) \(name)(\(trackingId)) " +
            "readWrite device array reference on \(stream.device.name) " +
            "\(String(describing: Scalar.self))[\(buffer.count)]",
            categories: .dataAlloc)
    }
    
    //----------------------------------------
    // copy from buffer
    public init<Scalar>(copying buffer: UnsafeBufferPointer<Scalar>,
                        name: String) throws {
        self.name = name
        masterVersion = 0
        isReadOnly = false
        byteCount = buffer.count * MemoryLayout<Scalar>.size
        register(type: Scalar.self, count: buffer.count)

        diagnostic("\(createString) \(name)(\(trackingId)) " +
            "\(String(describing: Scalar.self))[\(buffer.count)]",
            categories: .dataAlloc)
        
        // copy the data
        let stream = _Streams.local.appThreadStream
        do {
            try syncLastMutatingStream(with: stream, isReadOnly: false)
            _ = try readWrite(type: Scalar.self, using: stream)
                .initialize(from: buffer)
        } catch {
            stream.reportDevice(error: error)
        }
    }
    
    //----------------------------------------
    // init from other TensorArray
    public init<Scalar>(type: Scalar.Type,
                        copying other: TensorArray,
                        using stream: DeviceStream) throws {
        // initialize members
        isReadOnly = other.isReadOnly
        byteCount = other.byteCount
        name = other.name
        masterVersion = 0
        lastReadWriteStream = other.lastReadWriteStream
        register(type: UInt8.self, count: byteCount)
        
        // report
        diagnostic("\(createString) \(name)(\(trackingId)) init" +
            "\(setText(" copying ", color: .blue))" +
            "TensorArray(\(other.trackingId)) " +
            "\(String(describing: Scalar.self))" +
            "[\(byteCount / MemoryLayout<Scalar>.size)]",
            categories: [.dataAlloc, .dataCopy])

        // make sure there is something to copy
        guard let otherMaster = other.master else { return }
        
        // get the array replica for `stream`
        let replica = try getArray(type: Scalar.self, for: stream)
        replica.version = masterVersion
        
        // copy the other master array
        try syncLastMutatingStream(with: stream, isReadOnly: false)
        try replica.copyAsync(from: otherMaster, using: stream)
        lastReadWriteStream = stream

        diagnostic("\(copyString) \(name)(\(trackingId)) " +
            "\(otherMaster.device.name)" +
            "\(setText(" --> ", color: .blue))" +
            "\(stream.device.name)_s\(stream.id) " +
            "\(String(describing: Scalar.self))" +
            "[\(byteCount / MemoryLayout<Scalar>.size)]",
            categories: .dataCopy)
    }

    //----------------------------------------
    // syncLastMutatingStream
    // synchronizes the incoming stream with the last mutating stream
    private func syncLastMutatingStream(with stream: DeviceStream,
                                        isReadOnly: Bool) throws {
        // save if this stream mutates
        if !isReadOnly { lastReadWriteStream = stream }
        
        // if the streams are equal then they are already in order
        guard let last = lastReadWriteStream, last !== stream else { return }
        
        // if the streams are different, then sync
        try stream.sync(with: last, event: getSyncEvent(using: stream))
    }
    
    //----------------------------------------
    // object lifetime tracking for leak detection
    private func register<Scalar>(type: Scalar.Type, count: Int) {
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath, supplementalInfo:
                "\(String(describing: Scalar.self))[\(count)]")
    }
    
    //--------------------------------------------------------------------------
    deinit {
        // synchronize with all streams that have accessed these arrays
        // before freeing them
        do {
            try lastReadWriteStream?.blockCallerUntilComplete()
        } catch {
            writeLog(String(describing: error))
        }
        ObjectTracker.global.remove(trackingId: trackingId)

        if byteCount > 0 {
            diagnostic("\(releaseString) \(name)(\(trackingId)) ",
                categories: .dataAlloc)
        }
    }
    
    //--------------------------------------------------------------------------
    // getSyncEvent
    /// returns a reusable StreamEvent for stream synchronization
    private var _streamSyncEvent: StreamEvent!
    private func getSyncEvent(using stream: DeviceStream) throws -> StreamEvent{
        // lazy create when we have a stream to work with
        if _streamSyncEvent == nil {
            _streamSyncEvent = try stream.createEvent(options: [])
        }
        return _streamSyncEvent
    }

    //--------------------------------------------------------------------------
    // readOnly
    public func readOnly<Scalar>(type: Scalar.Type,
                                 using stream: DeviceStream) throws
        -> UnsafeBufferPointer<Scalar>
    {
        let buffer = try migrate(type: type, readOnly: true, using: stream)
        return UnsafeBufferPointer(buffer)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    public func readWrite<Scalar>(type: Scalar.Type,
                                  using stream: DeviceStream) throws ->
        UnsafeMutableBufferPointer<Scalar>
    {
        assert(!isReadOnly, "the TensorArray is read only")
        return try migrate(type: type, readOnly: false, using: stream)
    }

    //--------------------------------------------------------------------------
    /// migrate
    /// This migrates the master version of the data from wherever it is to
    /// the device associated with `stream` and returns a pointer to the data
    private func migrate<Scalar>(type: Scalar.Type,
                                 readOnly: Bool,
                                 using stream: DeviceStream) throws
        -> UnsafeMutableBufferPointer<Scalar>
    {
        // make sure the last operation is finished if it's on a
        // different stream
        try syncLastMutatingStream(with: stream, isReadOnly: readOnly)
        
        // get the array replica for `stream`
        let replica = try getArray(type: Scalar.self, for: stream)
        lastAccessCopiedBuffer = false

        // compare with master and copy if needed
        if let master = master, replica.version != master.version {
            // cross service?
            if replica.device.service.id != master.device.service.id {
                try copyCrossService(type: type, from: master, to: replica,
                                     using: stream)
                
            } else if replica.device.id != master.device.id {
                try copyCrossDevice(type: type, from: master, to: replica,
                                    using: stream)
            }
        }
        
        // set version
        if !readOnly { master = replica; masterVersion += 1 }
        replica.version = masterVersion
        return replica.buffer.bindMemory(to: Scalar.self)
    }

    //--------------------------------------------------------------------------
    // copyCrossService
    // copies from an array in one service to another
    private func copyCrossService<Scalar>(type: Scalar.Type,
                                          from master: DeviceArray,
                                          to other: DeviceArray,
                                          using stream: DeviceStream) throws
    {
        lastAccessCopiedBuffer = true
        
        if master.device.memoryAddressing == .unified {
            // copy host to discreet memory device
            if other.device.memoryAddressing == .discreet {
                let buffer = UnsafeRawBufferPointer(master.buffer)
                try other.copyAsync(from: buffer, using: stream)

                diagnostic("\(copyString) \(name)(\(trackingId)) " +
                    "uma:\(master.device.name)" +
                    "\(setText(" --> ", color: .blue))" +
                    "\(other.device.name)_s\(stream.id) " +
                    "\(String(describing: Scalar.self))" +
                    "[\(master.buffer.bindMemory(to: Scalar.self).count)]",
                    categories: .dataCopy)
            }
            // otherwise they are both unified, so do nothing
        } else if other.device.memoryAddressing == .unified {
            // device to host
            try master.copyAsync(to: other.buffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(master.device.name)_s\(stream.id)" +
                "\(setText(" --> ", color: .blue))uma:\(other.device.name) " +
                "\(String(describing: Scalar.self))" +
                "[\(master.buffer.bindMemory(to: Scalar.self).count)]",
                categories: .dataCopy)

        } else {
            // both are discreet and not in the same service, so
            // transfer to host memory as an intermediate step
            let host = try getArray(type: Scalar.self,
                                    for: _Streams.local.appThreadStream)
            try master.copyAsync(to: host.buffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(master.device.name)_s\(stream.id)" +
                "\(setText(" --> ", color: .blue))\(other.device.name)" +
                " bytes[\(byteCount)]", categories: .dataCopy)

            
            let hostBuffer = UnsafeRawBufferPointer(host.buffer)
            try other.copyAsync(from: hostBuffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(other.device.name)" +
                "\(setText(" --> ", color: .blue))" +
                "\(master.device.name)_s\(stream.id) " +
                "\(String(describing: Scalar.self))" +
                "[\(other.buffer.bindMemory(to: Scalar.self).count)]",
                categories: .dataCopy)
        }
    }
    
    //--------------------------------------------------------------------------
    // copyCrossDevice
    // copies from one discreet memory device to the other
    private func copyCrossDevice<Scalar>(type: Scalar.Type,
                                         from master: DeviceArray,
                                         to other: DeviceArray,
                                         using stream: DeviceStream) throws
    {
        // only copy if the devices have discreet memory
        guard master.device.memoryAddressing == .discreet else { return }
        lastAccessCopiedBuffer = true
        try other.copyAsync(from: master, using: stream)
        
        diagnostic("\(copyString) \(name)(\(trackingId)) " +
            "\(master.device.name)" +
            "\(setText(" --> ", color: .blue))" +
            "\(stream.device.name)_s\(stream.id) " +
            "\(String(describing: Scalar.self))" +
            "[\(master.buffer.bindMemory(to: Scalar.self).count)]",
            categories: .dataCopy)
    }
    
    //--------------------------------------------------------------------------
    // getArray(stream:
    // This manages a dictionary of replicated device arrays indexed
    // by serviceId and id. It will lazily create a device array if needed
    private func getArray<Scalar>(
        type: Scalar.Type,
        for stream: DeviceStream) throws -> DeviceArray
    {
        // lookup array associated with this stream
        let key = stream.device.deviceArrayReplicaKey
        if let replica = replicas[key] {
            return replica
        } else {
            // create the replica device array
            let array = try stream.device.createArray(count: byteCount)
            diagnostic("\(allocString) \(name)(\(trackingId)) " +
                "device array on \(stream.device.name) " +
                "\(String(describing: Scalar.self))" +
                "[\(array.buffer.bindMemory(to: Scalar.self).count)]",
                categories: .dataAlloc)
            
            array.version = -1
            replicas[key] = array
            return array
        }
    }
}
