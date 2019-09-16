//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==========================================================================
public protocol TensorIndexing: Strideable {
    associatedtype Position
    /// the linear buffer element index
    var bufferIndex: Int { get }
    
    /// initializer for starting at any position
    init<T>(view: T, at position: Position) where T: TensorView
    /// initializer specifically for the endIndex
    init<T>(endOf view: T) where T: TensorView
    
    /// highest frequency function to move the index
    /// use advanced(by n: for jumps or negative movement
    func increment() -> Self
}

public extension TensorIndexing {
    // Equatable
    @inlinable @inline(__always)
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.bufferIndex == rhs.bufferIndex
    }
    
    // Comparable
    @inlinable @inline(__always)
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.bufferIndex < rhs.bufferIndex
    }
    
    @inlinable @inline(__always)
    func distance(to other: Self) -> Int {
        return other.bufferIndex - bufferIndex
    }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View>: RandomAccessCollection
    where View: TensorView
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
    public subscript(index: View.Index) -> View.Element {
        return buffer[index.bufferIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView
{
    // properties
    public let buffer: UnsafeMutableBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) throws {
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
    public subscript(index: View.Index) -> View.Element {
        get {
            return buffer[index.bufferIndex]
        }
        set {
            buffer[index.bufferIndex] = newValue
        }
    }
}
