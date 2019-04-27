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
    /// Replica
    /// A data replica is maintained for each stream device
    class Replica {
        /// the array in the device address space
        let array: DeviceArray
        /// the last stream used to access the array. Used for synchronization
        /// and `deinit` of the array
        var lastStream: DeviceStream
        
        public init(array: DeviceArray, lastStream: DeviceStream) {
            self.array = array
            self.lastStream = lastStream
        }
    }
    
    //--------------------------------------------------------------------------
    /// used by TensorViews to synchronize access to this object
    public let accessQueue = DispatchQueue(label: "TensorArray.accessQueue")
    /// the size of the data array in bytes
    public let count: Int
    /// `true` if the data array points to an existing read only buffer
    public let isReadOnlyReference: Bool
    /// testing: `true` if the last access caused the contents of the
    /// buffer to be copied
    public private(set) var lastAccessCopiedBuffer = false
    /// testing: is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied. It's stored here instead of on the
    /// view, because the view is immutable when taking a read only pointer
    public var lastAccessMutatedView: Bool = false
    // whenever a buffer write pointer is taken, the associated DeviceArray
    // becomes the master copy for replication. Synchronization across threads
    // is still required for taking multiple write pointers, however
    // this does automatically synchronize data migrations.
    // The value will be `nil` if no access has been taken yet
    private var master: Replica?
    // this is incremented each time a write pointer is taken
    // all replicated buffers will stay in sync with this version
    private var masterVersion = -1
    /// name label used for logging
    public var name: String = ""
    /// replication collection
    private var replicas = [DeviceArrayReplicaKey : Replica]()
    /// the object tracking id
    public private(set) var trackingId = 0

    //--------------------------------------------------------------------------
    // initializers
    public init() {
        count = 0
        isReadOnlyReference = false
    }
    
    //----------------------------------------
    // create new array based on scalar size
    public init<Scalar>(type: Scalar.Type, count: Int) {
        isReadOnlyReference = false
        self.count = count * MemoryLayout<Scalar>.size
        register()
    }

    //----------------------------------------
    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a read only data object
    public init(referenceTo buffer: UnsafeRawBufferPointer) {
        // store
        masterVersion = 0
        isReadOnlyReference = true
        count = buffer.count
        let stream = _Streams.local.appThreadStream

        // create the replica device array
        diagnostic("\(referenceString) \(name)(\(trackingId)) " +
            "device array readOnly reference on \(stream.device.name) " +
            "bytes[\(count)]", categories: .dataAlloc)

        let key = stream.device.deviceArrayReplicaKey
        let array = stream.device.createReferenceArray(buffer: buffer)
        array.version = -1
        replicas[key] = Replica(array: array, lastStream: stream)

        // register self with object tracker
        register()
    }
    
    //----------------------------------------
    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a read only data object
    public init(referenceTo buffer: UnsafeMutableRawBufferPointer) {
        // store
        masterVersion = 0
        isReadOnlyReference = false
        count = buffer.count
        let stream = _Streams.local.appThreadStream
        
        // create the replica device array
        diagnostic("\(referenceString) \(name)(\(trackingId)) " +
            "device array readWrite reference on \(stream.device.name) " +
            "bytes[\(count)]", categories: .dataAlloc)
        
        let key = stream.device.deviceArrayReplicaKey
        let array = stream.device.createMutableReferenceArray(buffer: buffer)
        array.version = -1
        replicas[key] = Replica(array: array, lastStream: stream)
        
        // register self with object tracker
        register()
    }
    
    //----------------------------------------
    // copy from buffer
    public init(copying buffer: UnsafeRawBufferPointer) {
        masterVersion = 0
        isReadOnlyReference = false
        count = buffer.count
        
        // copy the data
        let stream = _Streams.local.appThreadStream
        do {
            _ = try readWrite(using: stream)
                .initializeMemory(as: UInt8.self, from: buffer)
        } catch {
            stream.reportDevice(error: error)
        }
        register()
    }
    
    //----------------------------------------
    // init from other TensorArray
    public init(copying other: TensorArray,
                using stream: DeviceStream?) throws {
        fatalError()
    }
    
    //----------------------------------------
    // object lifetime tracking for leak detection
    private func register() {
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath,
                      supplementalInfo: "bytes[\(count)]")
        
        if count > 0 {
            diagnostic("\(createString) \(name)(\(trackingId)) " +
                "bytes[\(count)]", categories: .dataAlloc)
        }
    }
    
    //--------------------------------------------------------------------------
    deinit {
        // synchronize with all streams that have accessed these arrays
        // before freeing them
        do {
            for replica in replicas.values {
                try replica.lastStream.blockCallerUntilComplete()
            }
        } catch {
            writeLog(String(describing: error))
        }
        ObjectTracker.global.remove(trackingId: trackingId)

        if count > 0 {
            diagnostic("\(releaseString) \(name)(\(trackingId)) " +
                "bytes[\(count)]", categories: .dataAlloc)
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
    public func readOnly(using stream: DeviceStream) throws
        -> UnsafeRawBufferPointer {
            let buffer = try migrate(readOnly: true, using: stream)
            return UnsafeRawBufferPointer(buffer)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    public func readWrite(using stream: DeviceStream) throws ->
        UnsafeMutableRawBufferPointer {
            assert(!isReadOnlyReference)
            return try migrate(readOnly: false, using: stream)
    }

    //--------------------------------------------------------------------------
    /// migrate
    /// This migrates the master version of the data from wherever it is to
    /// the device associated with `stream` and returns a pointer to the data
    private func migrate(readOnly: Bool, using stream: DeviceStream) throws
        -> UnsafeMutableRawBufferPointer
    {
        // get the array replica for `stream`
        let replica = try getArray(for: stream)
        
        // compare with master and copy if needed
        if let master = master, replica.array.version != master.array.version {
            // cross service?
            if replica.array.device.service.id != master.array.device.service.id {
                try copyCrossService(from: master, to: replica, using: stream)
                
            } else if replica.array.device.id != master.array.device.id {
                try copyCrossDevice(from: master, to: replica, using: stream)
            }
            lastAccessCopiedBuffer = true
        }
        
        // set version
        if !readOnly { master = replica; masterVersion += 1 }
        replica.array.version = masterVersion
        return replica.array.buffer
    }

    //--------------------------------------------------------------------------
    // copyCrossService
    // copies from an array in one service to another
    private func copyCrossService(from master: Replica, to other: Replica,
                                  using stream: DeviceStream) throws {
        let masterDevice = master.array.device
        let otherDevice = other.array.device
        
        if masterDevice.memoryAddressing == .unified {
            // copy host to discreet memory device
            if otherDevice.memoryAddressing == .discreet {
                let buffer = UnsafeRawBufferPointer(master.array.buffer)
                try other.array.copyAsync(from: buffer, using: stream)

                diagnostic("\(copyString) \(name)(\(trackingId)) " +
                    "\(otherDevice.name)" +
                    "\(setText(" --> ", color: .blue))" +
                    "\(masterDevice.name)_s\(stream.id) bytes[\(count)]",
                    categories: .dataCopy)
            }
            // otherwise they are both unified, so do nothing
        } else if otherDevice.memoryAddressing == .unified {
            // device to host
            try master.array.copyAsync(to: other.array.buffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(masterDevice.name)_s\(master.lastStream.id)" +
                "\(setText(" --> ", color: .blue))\(otherDevice.name)" +
                " bytes[\(count)]", categories: .dataCopy)

        } else {
            // both are discreet and not in the same service, so
            // transfer to host memory as an intermediate step
            let host = try getArray(for: _Streams.local.appThreadStream)
            try master.array.copyAsync(to: host.array.buffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(masterDevice.name)_s\(master.lastStream.id)" +
                "\(setText(" --> ", color: .blue))\(otherDevice.name)" +
                " bytes[\(count)]", categories: .dataCopy)

            
            let hostBuffer = UnsafeRawBufferPointer(host.array.buffer)
            try other.array.copyAsync(from: hostBuffer, using: stream)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(otherDevice.name)" +
                "\(setText(" --> ", color: .blue))" +
                "\(masterDevice.name)_s\(stream.id) bytes[\(count)]",
                categories: .dataCopy)
        }
    }
    
    //--------------------------------------------------------------------------
    // copyCrossDevice
    // copies from one discreet memory device to the other
    private func copyCrossDevice(from master: Replica, to other: Replica,
                                 using stream: DeviceStream) throws {
        // only copy if the devices have discreet memory
        guard master.array.device.memoryAddressing == .discreet else { return }
        try other.array.copyAsync(from: master.array, using: stream)
        
        diagnostic("\(copyString) \(name)(\(trackingId)) " +
            "\(master.lastStream.device.name)" +
            "\(setText(" --> ", color: .blue))" +
            "\(stream.device.name)_s\(stream.id) " +
            "bytes[\(count)]",
            categories: .dataCopy)
    }
    
    //--------------------------------------------------------------------------
    // getArray(stream:
    // This manages a dictionary of replicated device arrays indexed
    // by serviceId and id. It will lazily create a device array if needed
    private func getArray(for stream: DeviceStream) throws -> Replica {
        // lookup array associated with this stream
        let key = stream.device.deviceArrayReplicaKey
        if let replica = replicas[key] {
            // sync the requesting stream with the last stream that accessed it
            try stream.sync(with: replica.lastStream,
                            event: getSyncEvent(using: stream))
            // update the last stream used to access this array for sync
            replica.lastStream = stream
            return replica
            
        } else {
            // create the replica device array
            diagnostic("\(allocString) \(name)(\(trackingId)) " +
                "device array on \(stream.device.name) bytes[\(count)]",
                categories: .dataAlloc)
            
            let array = try stream.device.createArray(count: count)
            array.version = -1
            let replica = Replica(array: array, lastStream: stream)
            replicas[key] = replica
            return replica
        }
    }
}
