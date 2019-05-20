//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizing
/// enables quantized tensor indexing
public protocol Quantizing where Self: TensorView, Element: Quantizable {
    /// the type presented by the Values and MutableValues collections
    associatedtype Viewed: Quantizable
    /// the type of read only stored elements collection
    associatedtype ElementValues: RandomAccessCollection
        where ElementValues.Element == Element
    /// the type of read write stored elements collection
    associatedtype MutableElementValues:
        RandomAccessCollection & MutableCollection
        where MutableElementValues.Element == Element

    /// bias applied to Viewed value
    var bias: Viewed { get set }
    /// scale applied to Viewed value
    var scale: Viewed { get set }
    /// converts the tensor element value type to the viewed value type
    func convert(element: Element) -> Viewed
    /// converts the tensor viewed value type to the element value type
    func convert(viewed: Viewed) -> Element
    /// a collection of the stored quantized elements.
    /// Primarily intended for serialization
    func elementValues(using stream: DeviceStream?) throws -> ElementValues
    /// a mutable collection of the stored quantized elements
    /// Primarily intended for serialization
    mutating func elementMutableValues(using stream: DeviceStream?) throws
        -> MutableElementValues
}

public extension Quantizing {
    //--------------------------------------------------------------------------
    /// converts the tensor element value type to the viewed value type
    func convert(element: Element) -> Viewed {
        return Viewed(value: element, scale: scale, bias: bias)
    }

    //--------------------------------------------------------------------------
    /// converts the tensor viewed value type to the element value type
    func convert(viewed: Viewed) -> Element {
        return Element(value: viewed, scale: scale, bias: bias)
    }
    
    //--------------------------------------------------------------------------
    /// returns a collection of read only values
    func values(using stream: DeviceStream?) throws
        -> QTensorValueCollection<Self>
    {
        let buffer = try readOnly(using: stream)
        return try QTensorValueCollection(view: self, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
    /// returns a collection of read write values
    mutating func mutableValues(using stream: DeviceStream?) throws
        -> QTensorMutableValueCollection<Self>
    {
        let buffer = try readWrite(using: stream)
        return try QTensorMutableValueCollection(view: &self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
    /// returns a collection of read only values
    func elementValues(using stream: DeviceStream? = nil) throws
        -> TensorValueCollection<Self>
    {
        let buffer = try readOnly(using: stream)
        return try TensorValueCollection(view: self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
    /// returns a collection of read write values
    mutating func elementMutableValues(using stream: DeviceStream? = nil) throws
        -> TensorMutableValueCollection<Self>
    {
        let buffer = try readWrite(using: stream)
        return try TensorMutableValueCollection(view: &self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
    /// an array of viewed elements
    func elementArray() throws -> [Element] {
        return [Element](try elementValues())
    }
}

//==============================================================================
/// QTensorValueCollection
public struct QTensorValueCollection<View>: RandomAccessCollection
    where View: TensorView & Quantizing
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.shape.elementCount
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Viewed {
        return view.convert(element: buffer[index.dataIndex])
    }
}

//==============================================================================
/// QTensorMutableValueCollection
public struct QTensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView & Quantizing
{
    // properties
    public let view: View
    public let buffer: UnsafeMutableBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.shape.elementCount
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Viewed {
        get {
            return view.convert(element: buffer[index.dataIndex])
        }
        set {
            buffer[index.dataIndex] = view.convert(viewed: newValue)
        }
    }
}
