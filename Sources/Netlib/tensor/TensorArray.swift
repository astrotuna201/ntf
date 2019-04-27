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
    struct Replica {
        /// the array in the device address space
        let array: DeviceArray
        /// the last stream used to access the array. Used for synchronization
        /// and `deinit` of the array
        public var lastStream: DeviceStream
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
    public init(readOnlyReferenceTo buffer: UnsafeRawBufferPointer) {
        // store
        masterVersion = 0
        isReadOnlyReference = true
        count = buffer.count
//        let stream = _Streams.local.applicationThreadStream

        register()
    }
    
    //----------------------------------------
    // copy from buffer
    public init(buffer: UnsafeRawBufferPointer) {
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
    public init(withContentsOf other: TensorArray,
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
//        do {
//            // synchronize with all streams that have accessed these arrays
//            // before freeing them
//            for sid in 0..<deviceArrays.count {
//                for devId in 0..<deviceArrays[sid].count {
//                    if let info = deviceArrays[sid][devId] {
//                        try info.stream.blockCallerUntilComplete()
//                    }
//                }
//            }
//        } catch {
//            writeLog(String(describing: error))
//        }
//        ObjectTracker.global.remove(trackingId: trackingId)
//
//        if byteCount > 0 {
//            diagnostic("\(releaseString) \(name)(\(trackingId)) " +
//                "elements[\(elementCount)]", categories: .dataAlloc)
//        }
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
        // determine addressing combination
        fatalError()
//        let srcUsesUMA = master?.stream.device.memoryAddressing != .discreet
//        let dstUsesUMA = stream?.device.memoryAddressing != .discreet
//
//        // reset, this is to support automated tests
//        lastAccessCopiedBuffer = false
//
//        if srcUsesUMA {
//            if dstUsesUMA {
//                try setDeviceDataPointerToHostBuffer(readOnly: readOnly)
//            } else {
//                assert(stream != nil, streamRequired)
//                try host2device(readOnly: readOnly, using: stream!)
//            }
//        } else {
//            if dstUsesUMA {
//                try device2host(readOnly: readOnly)
//            } else {
//                assert(stream != nil, streamRequired)
//                try device2device(readOnly: readOnly, using: stream!)
//            }
//        }
    }

}
