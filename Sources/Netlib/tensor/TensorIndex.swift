//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorIndexing
public protocol TensorIndexing where Self: TensorView {
    /// returns a collection of values in spatial order
    func values() throws -> TensorValueCollection<Self>
    /// returns a collection of mutable values in spatial order
    mutating func mutableValues() throws -> TensorMutableValueCollection<Self>
}

extension TensorIndexing {
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order
    /// called to synchronize with the app thread
    func values() throws -> TensorValueCollection<Self> {
        return try TensorValueCollection(view: self, buffer: readOnly())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of mutable values in spatial order
    mutating func mutableValues() throws -> TensorMutableValueCollection<Self> {
        return try TensorMutableValueCollection(view: &self, buffer: readWrite())
    }

    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order as an array
    func array() throws -> [Scalar] {
        return try [Scalar](values())
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of read only values in spatial order
    func values(using stream: DeviceStream) throws
        -> TensorValueCollection<Self>
    {
        return try TensorValueCollection(view: self,
                                         buffer: readOnly(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of mutable values in spatial order
    mutating func mutableValues(using stream: DeviceStream) throws
        -> TensorMutableValueCollection<Self>
    {
        return try TensorMutableValueCollection(
            view: &self, buffer: readWrite(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    func value(at position: ViewPosition) throws -> Scalar {
        let buffer = try readOnly()
        let index = createIndex(at: position)
        return index.isPad ? padValue : buffer[index.dataIndex]
    }

    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    mutating func set(value: Scalar, at position: ViewPosition) throws {
        let buffer = try readWrite()
        let index = createIndex(at: position)
        buffer[index.dataIndex] = value
    }
}

//==========================================================================
public protocol TensorIndex: Strideable {
    // types
    typealias AdvanceFn = (Self, _ by: Int) -> Self

    // properties
    var advanceFn: AdvanceFn { get }
    var dataIndex: Int { get }
    var isPad: Bool { get }
}

public extension TensorIndex {
    // Equatable
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.dataIndex == rhs.dataIndex
    }

    // Comparable
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.dataIndex < rhs.dataIndex
    }

    // Strideable
    func advanced(by n: Int) -> Self {
        return advanceFn(self, n)
    }

    func distance(to other: Self) -> Int {
        return other.dataIndex - dataIndex
    }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View> where View: TensorView {
    public typealias Index = View.ViewIndex
    public typealias Scalar = View.Scalar
    
    // properties
    let buffer: UnsafeBufferPointer<View.Scalar>
    let startIndex: Index
    let endIndex: Index
    let count: Int
    let padValue: Scalar

    public init(view: View, buffer: UnsafeBufferPointer<Scalar>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
    }

    //--------------------------------------------------------------------------
    // Collection
    public func index(before i: Index) -> Index {
        return i.advanced(by: -1)
    }
    
    public func index(after i: Index) -> Index {
        return i.advanced(by: 1)
    }
    
    public subscript(index: Index) -> Scalar {
        return index.isPad ? padValue : buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View> where View: TensorView {
    public typealias Index = View.ViewIndex
    public typealias Scalar = View.Scalar
    
    // properties
    let buffer: UnsafeMutableBufferPointer<Scalar>
    let startIndex: Index
    let endIndex: Index
    let count: Int
    let padValue: Scalar
    
    public init(view: inout View, buffer: UnsafeMutableBufferPointer<Scalar>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
    }
    
    //--------------------------------------------------------------------------
    // Collection
    public func index(before i: Index) -> Index {
        return i.advanced(by: -1)
    }
    
    public func index(after i: Index) -> Index {
        return i.advanced(by: 1)
    }
    
    public subscript(index: Index) -> Scalar {
        return index.isPad ? padValue : buffer[index.dataIndex]
    }
}
