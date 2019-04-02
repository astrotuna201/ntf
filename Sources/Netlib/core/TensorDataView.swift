//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorDataView
public protocol TensorDataView: AnyScalar, Logging, Equatable {
    // types
    associatedtype Scalar: AnyScalar

    // properties
    var isContiguous: Bool { get }
    var isEmpty: Bool { get }
    var lastAccessMutated: Bool { get }
    var logging: LogInfo? { get set }
    var name: String { get set }
    var rank: Int { get }
    var shape: DataShape { get }
    
    init<T>(shapedLike other: T) where T: TensorDataView
}

public extension TensorDataView {
    var isContiguous: Bool { return shape.isContiguous }
    var isEmpty: Bool { return shape.isEmpty }
    var rank: Int { return shape.rank }
}

public extension TensorDataView where Self: TensorDataViewImpl {
    var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }
    
    func scalarValue() throws -> Scalar {
        assert(shape.elementCount == 1)
        return try ro()[0]
    }
}

//==============================================================================
// TensorDataViewImpl
public protocol TensorDataViewImpl: TensorDataView {
    // properties
    var isShared: Bool { get set }
    var lastAccessMutated: Bool { get set }
    var logging: LogInfo? { get set }
    var name: String { get set }
    var shape: DataShape { get set }
    var tensorData: TensorData { get set }
    var viewOffset: Int { get set }
    var viewByteOffset: Int { get }
    var viewSpanByteCount: Int { get }

    /// determines if the view holds a unique reference to the underlying
    /// TensorData array
    mutating func isUniqueReference() -> Bool
}

public extension TensorDataViewImpl {
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }
    
    var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }
    
    var viewByteOffset: Int { return viewOffset * MemoryLayout<Scalar>.size }
    var viewSpanByteCount: Int { return shape.elementSpanCount * MemoryLayout<Scalar>.size }

    //--------------------------------------------------------------------------
    // Equal values
    static func == (lhs: Self, rhs: Self) -> Bool {
        if lhs.tensorData === rhs.tensorData {
            // If they both reference the same tensorData then compare the views
            return lhs.viewOffset == rhs.viewOffset && lhs.shape == rhs.shape
            
        } else if lhs.shape.extents == rhs.shape.extents {
            // if the extents are equal then compare values
            // TODO use indexing
            fatalError("Not implemented")
        } else {
            return false
        }
    }
    
    //--------------------------------------------------------------------------
    // Equal references
    static func === (lhs: Self, rhs: Self) -> Bool {
        return lhs.tensorData === rhs.tensorData && lhs == rhs
    }
    

    //--------------------------------------------------------------------------
    // copyIfMutates
    //  Note: this should be called from inside the accessQueue.sync block
    mutating func copyIfMutates(using stream: DeviceStream? = nil) throws {
        // for unit tests
        lastAccessMutated = false
        guard !isShared && !isUniqueReference() else { return }
        
        lastAccessMutated = true
        if willLog(level: .diagnostic) == true {
            diagnostic("""
                \(mutationString) \(logging?.namePath ?? "")
                (\(tensorData.trackingId))  elements: \(shape.elementCount)
                """, categories: [.dataCopy, .dataMutation])
        }
        
        tensorData = try TensorData(withContentsOf: tensorData, using: stream)
    }
    
    //--------------------------------------------------------------------------
    // Read only buffer access
    func ro() throws -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            let buffer = try tensorData.roHostRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    // this version is for accelerator APIs
    func ro(using stream: DeviceStream) throws -> UnsafeRawPointer {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            let buffer = try tensorData.roDevicePointer(using: stream)
            return buffer.advanced(by: viewOffset)
        }
    }
    
    //--------------------------------------------------------------------------
    // Read Write buffer access
    mutating func rw() throws -> UnsafeMutableBufferPointer<Scalar> {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            try copyIfMutates()
            let buffer = try tensorData.rwHostMutableRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    // this version is for accelerator APIs
    mutating func rw(using stream: DeviceStream) throws -> UnsafeMutableRawPointer {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            let buffer = try tensorData.rwDevicePointer(using: stream)
            return buffer.advanced(by: viewOffset)
        }
    }
}
