//******************************************************************************
//  Created by Edward Connell on 4/17/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    func values() throws -> TensorViewCollection<Self> {
        return try TensorViewCollection(view: self)
    }

    func mutableValues() throws -> TensorViewMutableCollection<Self> {
        return try TensorViewMutableCollection(view: self)
    }
}

//==============================================================================
/// TensorViewCollection
/// returns a readonly collection view of the underlying tensorData.
public struct TensorViewCollection<View>: Collection where View: TensorView {
    public typealias Scalar = View.Scalar
    // properties
    let view: View
    let buffer: UnsafeBufferPointer<Scalar>
    let padValue: Scalar
    
    public init(view: View) throws {
        self.view = view
        buffer = try view.readOnly()
        padValue = view.padValue ?? Scalar()
    }

    //--------------------------------------------------------------------------
    // Collection
    public var startIndex: TensorIndex {
        return TensorIndex(offset: view.viewOffset)
    }
    public var endIndex: TensorIndex {
        return TensorIndex(
            offset: view.viewOffset + view.shape.elementSpanCount)
    }

    public func index(after i: TensorIndex) -> TensorIndex {
        return i.next()
    }

    public subscript(index: TensorIndex) -> Scalar {
        return index.data < 0 ? padValue : buffer[index.data]
    }
}

//==============================================================================
/// TensorViewMutableCollection
/// returns a readWrite collection view of the underlying tensorData.
public struct TensorViewMutableCollection<View>: MutableCollection
where View: TensorView {
    public typealias Scalar = View.Scalar
    // properties
    var view: View
    let buffer: UnsafeMutableBufferPointer<Scalar>
    
    public init(view: View) throws {
        self.view = view
        buffer = try self.view.readWrite()
    }
    
    //--------------------------------------------------------------------------
    // Collection
    public var startIndex: TensorIndex {
        return TensorIndex(offset: view.viewOffset)
    }
    public var endIndex: TensorIndex {
        return TensorIndex(
            offset: view.viewOffset + view.shape.elementSpanCount)
    }
    
    public func index(after i: TensorIndex) -> TensorIndex {
        return i.next()
    }
    
    public subscript(index: TensorIndex) -> Scalar {
        get {
            return buffer[index.data]
        }
        set {
            buffer[index.data] = newValue
        }
    }
}

//==============================================================================
/// TensorIndex
public struct TensorIndex : Comparable {
    public var shape: Int
    public var data: Int
    
    public init(offset: Int) {
        shape = offset
        data = offset
    }
    
    public static func < (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        return lhs.shape < rhs.shape
    }
    
    public func next() -> TensorIndex {
        var nextIndex = self
        nextIndex.shape += 1
        nextIndex.data += 1
        return nextIndex
    }
}

