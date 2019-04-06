//******************************************************************************
//  Created by Edward Connell on 4/5/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// DataShape
extension DataShape: Sequence {
    public func makeIterator() -> NDExtentIterator {
        return NDExtentIterator(shape: self, extentIndex: 0,
                                startingTensorIndex: 0)
    }
}

//==============================================================================
// DataShapeIterator
// This is a recursive iterator that works it's way through N dimensions
public protocol DataShapeIterator: Sequence, IteratorProtocol {
    /// the index of the extent being iterated
    var _extentIndex: Int { get set }
    /// the current iterator tensor index
    var _currentTensorIndex: Int { get set }
    /// the tensorIndex just past the end of this range
    var _tensorIndexPastEnd: Int { get set }
    /// the shape being iterated
    var _shape: DataShape { get set }
    /// returns an iterator for the tensor data indices for the elements
    /// contained by this extent, which is the next dimension
    var tensorIndices: IndexSequence { get }

    /// initializer
    /// - Parameter shape: the shape being iterated
    /// - Parameter extentIndex: the extent being iterated
    /// - Parameter startingTensorIndex: the starting tensor index
    init(shape: DataShape, extentIndex: Int, startingTensorIndex: Int)
    
    /// makes a sequence iterator for the next dimension
    func makeIterator() -> Self
    
    /// return iterator for the next dimension
    mutating func next() -> Self?
}

public extension DataShapeIterator {
    /// makes a sequence iterator for the next dimension
    func makeIterator() -> Self {
        assert(_extentIndex + 1 < _shape.rank)
        return Self.init(shape: _shape,
                         extentIndex: _extentIndex + 1,
                         startingTensorIndex: _currentTensorIndex)
    }
    
    /// return iterator for the next dimension
    mutating func next() -> Self? {
        // if we pass the end then return nil
        if _currentTensorIndex == _tensorIndexPastEnd {
            return nil
        } else {
            let iter = Self.init(shape: _shape,
                                 extentIndex: _extentIndex,
                                 startingTensorIndex: _currentTensorIndex)
            _currentTensorIndex += _shape.strides[_extentIndex]
            return iter
        }
    }
    
    /// returns an iterator for the tensor data indices for the elements
    /// contained by this extent, which is the next dimension
    var tensorIndices: IndexSequence {
        assert(_extentIndex + 1 < _shape.rank)
        return IndexSequence(shape: _shape, dim: _extentIndex + 1,
                             offset: _currentTensorIndex)
    }
}

//==============================================================================
// NDExtentIterator
public struct NDExtentIterator: DataShapeIterator {
    // properties
    public var _currentTensorIndex: Int
    public var _extentIndex: Int
    public var _shape: DataShape
    public var _tensorIndexPastEnd: Int
    
    /// initializer
    public init(shape: DataShape, extentIndex: Int, startingTensorIndex: Int) {
        _shape = shape
        _extentIndex = extentIndex
        _currentTensorIndex = startingTensorIndex
        _tensorIndexPastEnd = startingTensorIndex +
            shape.extents[extentIndex] * shape.strides[extentIndex]
    }
    
}

//==============================================================================
// IndexSequence
public struct IndexSequence: Sequence {
    let dim: Int
    var offset: Int
    let shape: DataShape

    init(shape: DataShape, dim: Int, offset: Int) {
        self.dim = dim
        self.offset = offset
        self.shape = shape
    }
    
    public func makeIterator() -> IndexIterator {
        return IndexIterator(shape: shape, dim: dim, offset: offset)
    }
}


public struct IndexIterator: IteratorProtocol {
    let dim: Int
    let endOffset: Int
    var offset: Int
    let shape: DataShape

    init(shape: DataShape, dim: Int, offset: Int) {
        self.dim = dim
        self.endOffset = offset + shape.extents[dim] * shape.strides[dim]
        self.offset = offset
        self.shape = shape
    }

    public mutating func next() -> Int? {
        // if we pass the end then return nil
        if offset == endOffset {
            return nil
        } else {
            let value = offset
            offset += shape.strides[dim]
            return value
        }
    }
}

