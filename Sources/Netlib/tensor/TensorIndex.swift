//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorView Collection extensions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    /// This is not an efficient way to get values
    func value(at index: [Int]) throws -> Scalar {
        let buffer = try readOnly()
        let padded = shape.padded(with: padding)
        let tensorIndex = TensorIteratorIndex<Self>(self, padded, at: index)
        return tensorIndex.isPad ? padValue : buffer[tensorIndex.dataIndex]
    }
    
    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    /// This is not an efficient way to set values
    mutating func set(value: Scalar, at index: [Int]) throws {
        let buffer = try readWrite()
        let padded = shape.padded(with: padding)
        let tensorIndex = TensorIteratorIndex<Self>(self, padded, at: index)
        buffer[tensorIndex.dataIndex] = value
    }
}

//==============================================================================
/// TensorIndexing
public protocol TensorIndexing {
    associatedtype View
    
    func computeIndex()
}

public extension TensorIndexing {
    func computeIndex() {
        
    }
}

public extension TensorIndexing where View: VectorView {
    func computeIndex() {
        
    }
}

//==============================================================================
/// TensorDirectIndex
public struct TensorDirectIndex<T>: TensorIndexing, Strideable, Comparable
where T: TensorView
{
    // types
    public typealias Index = TensorDirectIndex<T>
    public typealias View = T

    // properties
    var dataIndex: Int = 0
    let dataShape: DataShape
    let padding: [Padding]
    var rank: Int { return spatialIndex.count }
    var spatialIndex: ContiguousArray<Int>
    var viewIndex: Int = 0
    let viewShape: DataShape

    // initializers
    init(_ tensorView: T, index: [Int]) {
        spatialIndex = ContiguousArray(index)
        dataShape = tensorView.dataShape
        padding = tensorView.padding
        viewShape = tensorView.shape.padded(with: tensorView.padding)
        computeIndex()
    }

    //--------------------------------------------------------------------------
    // Equatable
    public static func == (lhs: Index, rhs: Index) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    public static func < (lhs: Index, rhs: Index) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.viewIndex < rhs.viewIndex
    }
    
    public func increment() -> Index {
        let next = self
        return next
    }
    
    public func advanced(by n: Int) -> Index {
        let next = self
        return next
    }
    
    public func distance(to other: Index) -> Int {
        return other.viewIndex - viewIndex
    }
}

