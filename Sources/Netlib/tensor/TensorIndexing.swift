//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==========================================================================
public protocol TensorIndex: Strideable {
    associatedtype Position
    /// the linear spatial position in the virtual view shape. The view
    /// shape and dataShape are equal if there is no padding or repeating
    var viewIndex: Int { get }
    /// the linear buffer index of the data corresponding to the `viewIndex`
    var dataIndex: Int { get }
    /// `true` if the index references a position in a padding region
    var isPad: Bool { get }
    
    /// initializer for starting at any position
    init<T>(view: T, at position: Position) where T: TensorView
    /// initializer specifically for the endIndex
    init<T>(endOf view: T) where T: TensorView
    
    /// highest frequency function to move the index
    /// use advanced(by n: for jumps or negative movement
    func increment() -> Self
}

public extension TensorIndex {
    // Equatable
    @inlinable @inline(__always)
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    @inlinable @inline(__always)
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex < rhs.viewIndex
    }
    
    @inlinable @inline(__always)
    func distance(to other: Self) -> Int {
        return other.viewIndex - viewIndex
    }
}

//==============================================================================
/// ExtentBounds
public struct ExtentBounds {
    public let before: Int
    public let after: Int
    public let viewExtent: Int
    public let viewStride: Int
    public let dataExtent: Int
    public let dataStride: Int
}

public typealias TensorBounds = ContiguousArray<ExtentBounds>

public extension TensorView {
    //--------------------------------------------------------------------------
    /// used by indexing objects
    func createTensorBounds() -> TensorBounds {
        var bounds = TensorBounds()
        let padShape = shape.padded(with: padding)
        for dim in 0..<rank {
            let pad = getPadding(for: dim)
            bounds.append(ExtentBounds(before: pad.before,
                                     after: padShape.extents[dim] - pad.after,
                                     viewExtent: padShape.extents[dim],
                                     viewStride: padShape.strides[dim],
                                     dataExtent: dataShape.extents[dim],
                                     dataStride: dataShape.strides[dim]))
        }
        return bounds
    }
}

//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order
    /// called to synchronize with the app thread
    @inlinable @inline(__always)
    func values() throws -> TensorValueCollection<Self> {
        return try TensorValueCollection(view: self, buffer: readOnly())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues() throws -> TensorMutableValueCollection<Self> {
        return try TensorMutableValueCollection(view: &self, buffer: readWrite())
    }

    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order as an array
    @inlinable @inline(__always)
    func array() throws -> [Scalar] {
        return try [Scalar](values())
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of read only values in spatial order
    @inlinable @inline(__always)
    func values(using stream: DeviceStream) throws
        -> TensorValueCollection<Self>
    {
        return try TensorValueCollection(
            view: self, buffer: readOnly(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues(using stream: DeviceStream) throws
        -> TensorMutableValueCollection<Self>
    {
        return try TensorMutableValueCollection(
            view: &self, buffer: readWrite(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    @inlinable @inline(__always)
    func value(at position: ViewIndex.Position) throws -> Scalar {
        let buffer = try readOnly()
        let index = ViewIndex.init(view: self, at: position)
        return index.isPad ? padValue : buffer[index.dataIndex]
    }

    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Scalar, at position: ViewIndex.Position) throws {
        let buffer = try readWrite()
        let index = ViewIndex.init(view: self, at: position)
        buffer[index.dataIndex] = value
    }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View>: RandomAccessCollection
    where View: TensorView
{
    // properties
    public let buffer: UnsafeBufferPointer<View.Scalar>
    public let startIndex: View.ViewIndex
    public let endIndex: View.ViewIndex
    public let count: Int
    public let padValue: View.Scalar

    public init(view: View, buffer: UnsafeBufferPointer<View.Scalar>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
    }

    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.ViewIndex) -> View.ViewIndex {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.ViewIndex) -> View.ViewIndex {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.ViewIndex) -> View.Scalar {
        return index.isPad ? padValue : buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View>: RandomAccessCollection,
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
    @inlinable @inline(__always)
    public func index(before i: Index) -> Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: Index) -> Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: Index) -> Scalar {
        get {
            return index.isPad ? padValue : buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}
