//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch
import TensorFlow

public typealias BufferUInt8 = UnsafeBufferPointer<UInt8>
public typealias MutableBufferUInt8 = UnsafeMutableBufferPointer<UInt8>

public class TensorData<Scalar> : ObjectTracking, Logging
where Scalar: AnyScalar & TensorFlowScalar {
    //--------------------------------------------------------------------------
    // properties
    public let accessQueue = DispatchQueue(label: "TensorData.accessQueue")
    public let elementCount: Int
    public let elementSize = MemoryLayout<Scalar>.size
    public var byteCount: Int { return elementSize * elementCount }
    public var autoReleaseUmaBuffer = false

    // object tracking
    public private(set) var trackingId = 0
    public var logging: LogInfo?

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
    
    //-----------------------------------
    // The host array is the host device data array. It is used when doing
    // operations synced with the app thread.
    private var hostVersion = -1
    private var _hostArray: [Scalar]? = nil
//    private var hostArray: [Scalar] = {
//        if _hostArray == nil { createHostArray() }
//        return _hostArray!
////        return [Scalar]()
//    }()
    
    // The hostBuffer points to the host data used by this object. Usually it
    // will point to the hostArray, but it can also point to a read only
    // buffer specified during init. The purpose is to use data from something
    // like a memory mapped file without copying it.
    private var hostBuffer: UnsafeMutableRawBufferPointer!
    
    //-----------------------------------
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
    
    // Empty
    public convenience init() {
        self.init(logging: nil, elementCount: 0)
    }

    // All initializers retain the data except this one
    // which creates a read only reference to avoid unnecessary copying from
    // a read only data object
    public init(logging: LogInfo?, readOnlyReferenceTo buffer: UnsafeRawBufferPointer) {
        self.logging = logging
        isReadOnlyReference = true
        elementCount = buffer.count
        masterVersion = 0
        hostVersion = 0

        // we won't ever actually mutate in this case
        hostBuffer = UnsafeMutableRawBufferPointer(
            start: UnsafeMutableRawPointer(OpaquePointer(buffer.baseAddress)),
            count: buffer.count)
        register()
    }

    //----------------------------------------
    // copy from buffer
    public init(logging: LogInfo?, buffer: UnsafeBufferPointer<Scalar>) {
        self.logging = logging
        isReadOnlyReference = false
        self.elementCount = buffer.count
        do {
            _ = try rwHostBuffer().initialize(from: buffer)
        } catch {
            // TODO: what do we want to do here when it should never fail
        }
        assert(hostVersion == 0 && masterVersion == 0)
        register()
    }

    //----------------------------------------
    // create new space
    public init(logging: LogInfo?, elementCount: Int, name: String? = nil) {
        isReadOnlyReference = false
        self.logging = logging
        self.elementCount = elementCount
        self._name = name
        register()
    }

    //----------------------------------------
    // object lifetime tracking for leak detection
    private func register() {
        trackingId = ObjectTracker.global
            .register(self,
                      namePath: logging?.namePath,
                      supplementalInfo: "elementCount: \(elementCount)")

        if elementCount > 0 && willLog(level: .diagnostic) {
            diagnostic("\(createString) \(name)(\(trackingId)) " +
                    "elements: \(elementCount)", categories: .dataAlloc)
        }
    }

    //----------------------------------------
    deinit {
        do {
            // synchronize with all streams that have accessed these arrays
            // before freeing them
            for sid in 0..<deviceArrays.count {
                for devId in 0..<deviceArrays[sid].count {
                    if let info = deviceArrays[sid][devId] {
                        try info.stream.blockCallerUntilComplete()
                    }
                }
            }
        } catch {
            writeLog(String(describing: error))
        }
        ObjectTracker.global.remove(trackingId: trackingId)

        if elementCount > 0 && willLog(level: .diagnostic) {
            diagnostic("\(releaseString) \(name)(\(trackingId)) " +
                "elements: \(elementCount)", categories: .dataAlloc)
        }
    }

    //----------------------------------------
    // init from other TensorData
    public init(withContentsOf other: TensorData,
                using stream: DeviceStream? = nil) throws {
        // init
        isReadOnlyReference = other.isReadOnlyReference
        elementCount = other.elementCount
        name = other.name
        masterVersion = 0
        hostVersion = masterVersion
        register()
        
        if willLog(level: .diagnostic) {
            let streamIdStr = stream == nil ? "nil" : "\(stream!.id)"
            diagnostic("\(createString) \(name)(\(trackingId)) init" +
                "\(setText(" copying ", color: .blue))" +
                "DataArray(\(other.trackingId)) elements: \(other.elementCount) " +
                "stream id(\(streamIdStr))", categories: [.dataAlloc, .dataCopy])
        }
        
        if isReadOnlyReference {
            // point to external data buffer, such as LMDB memory mapped data record
            assert(master == nil)
            hostBuffer = other.hostBuffer
            
        } else if let stream = stream {
            // get new array for the target stream's device location
            let arrayInfo = try getArray(for: stream)
            let array     = arrayInfo.array
            array.version = masterVersion
            
            if let otherMaster = other.master {
                // sync streams and copy
                try stream.sync(with: otherMaster.stream,
                                event: getSyncEvent(using: stream))
                try array.copyAsync(from: otherMaster.array, using: stream)
                
            } else {
                // uma to device
                try array.copyAsync(from: other.roHostRawBuffer(), using: stream)
            }
            
            // set the master
            master = arrayInfo
            
        } else {
            // get pointer to this array's umaBuffer
            let buffer = try rwHostMutableRawBuffer()
            
            if let otherMaster = other.master {
                // synchronous device to umaArray
                try otherMaster.array.copy(to: buffer, using: otherMaster.stream)
                
            } else {
                // umaArray to umaArray
                try buffer.copyMemory(from: other.roHostRawBuffer())
            }
        }
    }
    
    //--------------------------------------------------------------------------
    // ro
    public func roHostBuffer() throws -> UnsafeBufferPointer<Scalar> {
        try migrate(readOnly: true)
        return UnsafeBufferPointer(hostBuffer.bindMemory(to: Scalar.self))
    }

    public func roHostRawBuffer() throws -> UnsafeRawBufferPointer {
        try migrate(readOnly: true)
        return UnsafeRawBufferPointer(hostBuffer)
    }
    
    public func roDevicePointer(using stream: DeviceStream) throws -> UnsafeRawPointer {
        try migrate(readOnly: true, using: stream)
        return UnsafeRawPointer(deviceDataPointer)
    }

    //--------------------------------------------------------------------------
    // rw
    public func rwHostBuffer() throws -> UnsafeMutableBufferPointer<Scalar> {
        assert(!isReadOnlyReference)
        try migrate(readOnly: false)
        return hostBuffer.bindMemory(to: Scalar.self)
    }

    public func rwHostMutableRawBuffer() throws -> UnsafeMutableRawBufferPointer {
        assert(!isReadOnlyReference)
        try migrate(readOnly: false)
        return UnsafeMutableRawBufferPointer(hostBuffer)
    }

    public func rwDevicePointer(using stream: DeviceStream) throws ->
        UnsafeMutableRawPointer {
        assert(!isReadOnlyReference)
        try migrate(readOnly: false, using: stream)
        return deviceDataPointer
    }

    //--------------------------------------------------------------------------
    // migrate
    // This migrates
    private func migrate(readOnly: Bool, using stream: DeviceStream? = nil) throws {
        // if the array is empty then there is nothing to do
        guard !isReadOnlyReference && elementCount > 0 else { return }
        let srcUsesUMA = master?.stream.device.usesUnifiedAddressing ?? true
        let dstUsesUMA = stream?.device.usesUnifiedAddressing ?? true

        // reset, this is to support automated tests
        lastAccessCopiedBuffer = false

        if srcUsesUMA {
            if dstUsesUMA {
                try setDeviceDataPointerToHostBuffer(readOnly: readOnly)
            } else {
                assert(stream != nil, streamRequired)
                try host2device(readOnly: readOnly, using: stream!)
            }
        } else {
            if dstUsesUMA {
                try device2host(readOnly: readOnly)
            } else {
                assert(stream != nil, streamRequired)
                try device2device(readOnly: readOnly, using: stream!)
            }
        }
    }

    //--------------------------------------------------------------------------
    // getArray
    // This manages a dictionary of replicated device arrays indexed
    // by serviceId and id. It will lazily create a device array if needed
    private func getArray(for stream: DeviceStream) throws -> ArrayInfo {
        let device = stream.device
        let serviceId = device.service.id

        // add the device array list if needed
        if deviceArrays.count <= serviceId {
            let addCount = max(serviceId + 1, 2) - deviceArrays.count
            for _ in 0..<addCount {    deviceArrays.append([ArrayInfo?]()) }
        }

        // create array list if needed
        if deviceArrays[serviceId].isEmpty {
            deviceArrays[serviceId] =
                [ArrayInfo?](repeating: nil, count: device.service.devices.count)
        }

        // return existing if found
        if let info = deviceArrays[serviceId][device.id] {
            // sync the requesting stream with the last stream that accessed it
            try stream.sync(with: info.stream, event: getSyncEvent(using: stream))

            // update the last stream used to access this array for sync purposes
            info.stream = stream
            return info

        } else {
            // create the device array
            if willLog(level: .diagnostic) {
                diagnostic("\(createString) \(name)(\(trackingId)) " +
                    "allocating array on device(\(device.id)) elements: \(elementCount)",
                    categories: .dataAlloc)
            }
            let array = try device.createArray(count: byteCount)
            array.version = -1
            let info = ArrayInfo(array: array, stream: stream)
            deviceArrays[serviceId][device.id] = info
            return info
        }
    }

    //--------------------------------------------------------------------------
    // createHostArray
    private func createHostArray() throws {
        if willLog(level: .diagnostic) {
            diagnostic("\(createString) \(name)(\(trackingId)) " +
                "host array  elements: \(elementCount)", categories: .dataAlloc)
        }
        hostVersion = -1
        _hostArray = [Scalar](repeating: Scalar(any: 0), count: elementCount)
        hostBuffer = _hostArray!.withUnsafeMutableBytes { $0 }
    }

    //-----------------------------------
    // releaseHostArray
    private func releaseHostArray() {
        precondition(!isReadOnlyReference)
        if willLog(level: .diagnostic) {
            diagnostic(
                "\(releaseString) \(name) DataArray(\(trackingId)) host array " +
                "elements: \(elementCount)", categories: .dataAlloc)
        }
        _hostArray = nil
        hostBuffer = nil
    }

    //--------------------------------------------------------------------------
    // setDeviceDataPointerToHostBuffer
    private func setDeviceDataPointerToHostBuffer(readOnly: Bool) throws {
        assert(!isReadOnlyReference)
        // lazily create the uma buffer if needed
        if _hostArray == nil { try createHostArray() }
        deviceDataPointer = UnsafeMutableRawPointer(hostBuffer.baseAddress!)
        if !readOnly { master = nil; masterVersion += 1 }
        hostVersion = masterVersion
    }

    //--------------------------------------------------------------------------
    // host2device
    private func host2device(readOnly: Bool, using stream: DeviceStream) throws {
        let arrayInfo = try getArray(for: stream)
        let array     = arrayInfo.array
        deviceDataPointer = array.data

        if hostBuffer == nil {
            // clear the device buffer and set it to be the new master
            try array.zero(using: stream)
            master = arrayInfo

        } else if array.version != masterVersion {
            // copy host data to device if it exists and is needed
            if willLog(level: .diagnostic) {
                diagnostic("\(copyString) \(name)(\(trackingId)) host" +
                    "\(setText(" ---> ", color: .blue))" +
                    "d\(stream.device.id)_s\(stream.id) elements: \(elementCount)",
                    categories: .dataCopy)
            }

            try array.copyAsync(from: UnsafeRawBufferPointer(hostBuffer!),
                                using: stream)
            lastAccessCopiedBuffer = true

            if autoReleaseUmaBuffer && !isReadOnlyReference {
                // wait for the copy to complete, free the uma array,
                // and specify the device array as the new master
                try stream.blockCallerUntilComplete()
                releaseHostArray()
                master = arrayInfo
            }
        }

        // set version
        if !readOnly { master = arrayInfo; masterVersion += 1 }
        array.version = masterVersion
    }

    //--------------------------------------------------------------------------
    // device2host
    private func device2host(readOnly: Bool) throws {
        // master cannot be nil
        let master = self.master!
        assert(master.array.version == masterVersion)

        // lazily create the uma buffer if needed
        if hostBuffer == nil { try createHostArray() }
        deviceDataPointer = UnsafeMutableRawPointer(hostBuffer.baseAddress!)

        // copy if needed
        if hostVersion != masterVersion {
            if willLog(level: .diagnostic) {
                diagnostic("\(copyString) \(name)(\(trackingId)) " +
                    "d\(master.stream.device.id)_s\(master.stream.id)" +
                    "\(setText(" ---> ", color: .blue)) host" +
                    " elements: \(elementCount)", categories: .dataCopy)
            }

            // synchronous copy
            try master.array.copy(to: hostBuffer, using: master.stream)
            lastAccessCopiedBuffer = true
        }

        // set version
        if !readOnly { self.master = nil; masterVersion += 1 }
        hostVersion = masterVersion
    }

    //--------------------------------------------------------------------------
    // device2device
    private func device2device(readOnly: Bool, using stream: DeviceStream) throws {
        // master cannot be nil
        let master = self.master!
        assert(master.array.version == masterVersion)

        // get array for stream's device and set deviceBuffer pointer
        let arrayInfo = try getArray(for: stream)
        let array     = arrayInfo.array
        deviceDataPointer = array.data

        // synchronize output stream with master stream
        try stream.sync(with: master.stream, event: getSyncEvent(using: stream))

        // copy only if versions do not match
        if array.version != masterVersion {
            // copy within same service
            if master.stream.device.service.id == stream.device.service.id {
                // copy cross device within the same service if needed
                if master.stream.device.id != stream.device.id {
                    if willLog(level: .diagnostic) {
                        diagnostic("\(copyString) \(name)(\(trackingId)) " +
                            "device(\(master.stream.device.id))" +
                            "\(setText(" ---> ", color: .blue))" +
                            "device(\(stream.device.id)) elements: \(elementCount)",
                            categories: .dataCopy)
                    }
                    try array.copyAsync(from: master.array, using: stream)
                    lastAccessCopiedBuffer = true
                }

            } else {
                fatalError()
                //                if willLog(level: .diagnostic) == true {
                //                    diagnostic("\(copyString) \(name)(\(trackingId)) cross service from " +
                //                        "device(\(master.stream.device.id))" +
                //                    "\(setText(" ---> ", color: .blue))" +
                //                        "device(\(stream.device.id)) elementCount: \(elementCount)",
                //                        categories: .dataCopy)
                //                }
                //
                //                // cross service non-uma migration
                //                // copy data to uma buffer
                //                if umaBuffer == nil { try createHostArray() }
                //                try master.array.copy(to: umaBuffer, using: master.stream)
                //
                //                // copy data to destination device
                //                try dest.array.copyAsync(from: BufferUInt8(umaBuffer), using: stream)
                //
                //                if autoReleaseUmaBuffer {
                //                    // wait for the copy to complete, free the uma array,
                //                    // and specify the device array as the new master
                //                    try stream.blockCallerUntilComplete()
                //                    releaseHostArray()
                //                    self.master = dest
                //                }
                //
                //                lastAccessCopiedBuffer = true
            }
        }

        // set version
        if !readOnly { self.master = arrayInfo; masterVersion += 1 }
        self.master!.array.version = masterVersion
        array.version = masterVersion
    }
}
