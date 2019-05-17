//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizing
/// enables quantized tensor indexing
public protocol Quantizing where Self: TensorView {
    // types
    associatedtype Viewed
    
    // properties
    /// bias applied to Viewed value
    var bias: Float { get set }
    /// scale applied to Viewed value
    var scale: Float { get set }
    /// integer to float transform scale
    var _transformScale: Float { get set }
    /// float to integer transform scale
    var _inverseTransformScale: Float { get set }
    
    /// converts the tensor element value type to the viewed value type
    func convert(element: Element) -> Viewed
    /// converts the tensor viewed value type to the element value type
    func convert(viewed: Viewed) -> Element

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
    public let padValue: View.Viewed

    // initializers
    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.convert(element: view.padValue)
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
        return index.isPad ? padValue :
            view.convert(element: buffer[index.dataIndex])
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
    public let padValue: View.Viewed

    // initializers
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.convert(element: view.padValue)
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
            return index.isPad ? padValue :
                view.convert(element: buffer[index.dataIndex])
        }
        set {
            buffer[index.dataIndex] = view.convert(viewed: newValue)
        }
    }
}
