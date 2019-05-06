//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==========================================================================
public protocol TensorIndex: Strideable {
    // types
    typealias AdvanceFn = (Self, _ by: Int) -> Self

    // properties
    var advanceFn: AdvanceFn { get }
    var viewIndex: Int { get }
    var dataIndex: Int { get }
    var isPad: Bool { get }
}

public extension TensorIndex {
    // Equatable
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex < rhs.viewIndex
    }
    
    // Strideable
    func advanced(by n: Int) -> Self {
        return advanceFn(self, n)
    }
    
    func distance(to other: Self) -> Int {
        return other.viewIndex - viewIndex
    }
}

//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order
    /// called to synchronize with the app thread
    func values() throws -> TensorViewCollection<Self> {
        return try TensorViewCollection(view: self, buffer: readOnly())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of mutable values in spatial order
    mutating func mutableValues() throws -> TensorViewMutableCollection<Self> {
        return try TensorViewMutableCollection(view: &self, buffer: readWrite())
    }

    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order as an array
    func array() throws -> [Scalar] {
        return try [Scalar](values())
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of read only values in spatial order
    func values(using stream: DeviceStream) throws
        -> TensorViewCollection<Self>
    {
        return try TensorViewCollection(
            view: self, buffer: readOnly(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of mutable values in spatial order
    mutating func mutableValues(using stream: DeviceStream) throws
        -> TensorViewMutableCollection<Self>
    {
        return try TensorViewMutableCollection(
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

//==============================================================================
/// TensorViewCollection
public struct TensorViewCollection<View>: RandomAccessCollection
where View: TensorView
{
    public typealias Index = View.ViewIndex
    public typealias Scalar = View.Scalar
    
    // properties
    public let buffer: UnsafeBufferPointer<View.Scalar>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Scalar

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
/// TensorViewMutableCollection
public struct TensorViewMutableCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView
{
    public typealias Index = View.ViewIndex
    public typealias Scalar = View.Scalar
    
    // properties
    public let buffer: UnsafeMutableBufferPointer<Scalar>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Scalar
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<Scalar>) throws {
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
        get {
            return index.isPad ? padValue : buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}
