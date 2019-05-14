//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizing
/// performs quantized tensor indexing
public protocol Quantizing where
    Self: TensorView, Q.Stored == Stored, Q.Viewed == Viewed
{
    associatedtype Q: Quantizer
    
    var quantizer: Q { get }

    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         padding: [Padding]?,
         padValue: Stored?,
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         isShared: Bool,
         quantizer: Q,
         scalars: [Stored]?)
}

//==============================================================================
/// TensorView Quantizing extensions
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
    mutating func mutableValues() throws -> QTensorMutableValueCollection<Self>{
        return try QTensorMutableValueCollection(view: &self,
                                                 buffer: readWrite())
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
    func value(at position: Index.Position) throws -> Viewed {
        let buffer = try readOnly()
        let index = Index.init(view: self, at: position)
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return quantizer.convert(stored: stored)
    }
    
    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Viewed, at position: Index.Position) throws {
        let buffer = try readWrite()
        let index = Index.init(view: self, at: position)
        buffer[index.dataIndex] = quantizer.convert(viewed: value)
    }
}

//==============================================================================
/// QTensorValueCollection
public struct QTensorValueCollection<View>: RandomAccessCollection
    where View: TensorView & Quantizing
{
    // types
    public typealias Index = View.Index
    public typealias Stored = View.Stored
    public typealias Viewed = View.Viewed
    
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<Stored>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Stored
    
    // initializers
    public init(view: View, buffer: UnsafeBufferPointer<Stored>) throws {
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
    public subscript(index: Index) -> Viewed {
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return view.quantizer.convert(stored: stored)
    }
}

//==============================================================================
/// QTensorMutableValueCollection
public struct QTensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView & Quantizing
{
    // types
    public typealias Index = View.Index
    public typealias Stored = View.Stored
    public typealias Viewed = View.Viewed

    // properties
    public let view: View
    public let buffer: UnsafeMutableBufferPointer<Stored>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Stored
    
    // initializers
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<Stored>) throws {
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
    public subscript(index: Index) -> Viewed {
        get {
            let stored = index.isPad ? padValue : buffer[index.dataIndex]
            return view.quantizer.convert(stored: stored)
        }
        set {
            buffer[index.dataIndex] = view.quantizer.convert(viewed: newValue)
        }
    }
}
