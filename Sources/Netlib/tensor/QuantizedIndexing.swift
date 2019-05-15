//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// Quantizing
/// performs quantized tensor indexing
public protocol Quantizing where
    Self: TensorView, Q.Element == Element, Q.Viewed == Viewed
{
    associatedtype Q: Quantizer
    
    var quantizer: Q { get }

    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         padding: [Padding]?,
         padValue: Element?,
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         isShared: Bool,
         quantizer: Q,
         scalars: [Element]?)
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
    public let padValue: View.Element
    
    // initializers
    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) throws {
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
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Viewed {
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return view.quantizer.convert(stored: stored)
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
    public let padValue: View.Element
    
    // initializers
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) throws {
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
            let stored = index.isPad ? padValue : buffer[index.dataIndex]
            return view.quantizer.convert(stored: stored)
        }
        set {
            buffer[index.dataIndex] = view.quantizer.convert(viewed: newValue)
        }
    }
}
