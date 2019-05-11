//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizing TensorView Collection extensions
public extension TensorView where Self: Quantizing {
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order
    /// called to synchronize with the app thread
    @inlinable @inline(__always)
    func values() throws -> QTensorValueCollection<Self> {
        return try QTensorValueCollection(view: self, buffer: readOnly())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues() throws -> QTensorMutableValueCollection<Self> {
        return try QTensorMutableValueCollection(view: &self, buffer: readWrite())
    }

    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order as an array
    @inlinable @inline(__always)
    func array() throws -> [Viewed] {
        return try [Viewed](values())
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of read only values in spatial order
    @inlinable @inline(__always)
    func values(using stream: DeviceStream) throws
        -> QTensorValueCollection<Self>
    {
        return try QTensorValueCollection(
            view: self, buffer: readOnly(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues(using stream: DeviceStream) throws
        -> QTensorMutableValueCollection<Self>
    {
        return try QTensorMutableValueCollection(
            view: &self, buffer: readWrite(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    @inlinable @inline(__always)
    func value(at position: ViewIndex.Position) throws -> Viewed {
        let buffer = try readOnly()
        let index = ViewIndex.init(view: self, at: position)
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return quantizer.convert(stored: stored)
    }

    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Viewed, at position: ViewIndex.Position) throws {
        let buffer = try readWrite()
        let index = ViewIndex.init(view: self, at: position)
        buffer[index.dataIndex] = quantizer.convert(viewed: value)
    }
}

//==============================================================================
/// QTensorValueCollection
public struct QTensorValueCollection<View>: RandomAccessCollection
    where View: TensorView & Quantizing
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Scalar>
    public let startIndex: View.ViewIndex
    public let endIndex: View.ViewIndex
    public let count: Int
    public let padValue: View.Scalar

    public init(view: View, buffer: UnsafeBufferPointer<View.Scalar>) throws {
        self.view = view
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
    public subscript(index: View.ViewIndex) -> View.Viewed {
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return view.quantizer.convert(stored: stored)
    }
}

//==============================================================================
/// QTensorMutableValueCollection
public struct QTensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView & Quantizing
{
    public typealias Index = View.ViewIndex
    public typealias Scalar = View.Scalar
    
    // properties
    public let view: View
    public let buffer: UnsafeMutableBufferPointer<Scalar>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Scalar
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<Scalar>) throws {
        self.view = view
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
    public subscript(index: Index) -> View.Viewed {
        get {
            let stored = index.isPad ? padValue : buffer[index.dataIndex]
            return view.quantizer.convert(stored: stored)
        }
        set {
            buffer[index.dataIndex] = view.quantizer.convert(viewed: newValue)
        }
    }
}
