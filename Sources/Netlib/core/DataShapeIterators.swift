//******************************************************************************
//  Created by Edward Connell on 4/5/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// DataShapeSequenceIterable
// This is a recursive iterator that works it's way through N dimensions
public protocol DataShapeSequenceIterable: IteratorProtocol {
    /// the current position in nD space
    var currentPosition: [ExtentPosition]? { get set }
    /// the initial position in nD space
    var initialPosition: [ExtentPosition]? { get set }
    /// the index of the last dimension
    var lastDimension: Int { get set }
    /// fully specified initializer
    init()
}

public struct ExtentPosition {
    var position: Int
    let stride: Int
    let pastEnd: Int
}

public extension DataShapeSequenceIterable {
    //--------------------------------------------------------------------------
    /// initializer
    init(shape: DataShape, dimension: Int, startingIndex: Int) {
        self.init()
        guard !shape.isEmpty else { return }
        assert(dimension < shape.rank)
        lastDimension = shape.rank - dimension - 1
        initialPosition = []
        
        for dim in dimension..<shape.rank {
            // the initial end position along each dimension
            let span = shape.extents[dim] * shape.strides[dim]
            
            // record the starting point for each dimension
            initialPosition!.append(
                ExtentPosition(position: startingIndex,
                               stride: shape.strides[dim],
                               pastEnd: startingIndex + span))
        }
    }
    
    //--------------------------------------------------------------------------
    /// advances the lastDimension. If it can't, then `currentPosition`
    /// is set to `nil` this is a recursive function
    /// - Returns: the new position
    mutating func advancePosition(for dimension: Int) -> Int? {
        guard dimension >= 0 else {
            currentPosition = nil
            return nil
        }

        var nextPos: Int?
        if currentPosition == nil {
            nextPos = initialPosition == nil ? nil : 0
            currentPosition = initialPosition
        } else {
            // advance the position for this dimension by it's stride
            currentPosition![dimension].position
                += currentPosition![dimension].stride
            
            // if past the end then go back a dimension and advance
            if currentPosition![dimension].position ==
                currentPosition![dimension].pastEnd {
                // make a recursive call
                if let start = advancePosition(for: dimension - 1) {
                    currentPosition![dimension].position = start
                    nextPos = start
                }
            } else {
                nextPos = currentPosition![dimension].position
            }
        }
        return nextPos
    }
}

//==============================================================================
/// DataShapeSequenceIterator
/// This iterates the tensorData indexes described by
/// an N dimensional DataShape as a single linear Sequence
public struct DataShapeSequenceIterator: DataShapeSequenceIterable {
    public var currentPosition: [ExtentPosition]?
    public var initialPosition: [ExtentPosition]?
    public var lastDimension: Int = 0
    public init() {}
    
    public mutating func next() -> Int? {
        return advancePosition(for: lastDimension)
    }
}

//==============================================================================
// DataShapeSequence
public struct DataShapeSequence: Sequence {
    let shape: DataShape
    let dimension: Int
    let startingIndex: Int
    
    public func makeIterator() -> DataShapeSequenceIterator {
        return DataShapeSequenceIterator(shape: shape, dimension: dimension,
                                         startingIndex: startingIndex)
    }
}

extension DataShape {
    /// returns a Sequence of `tensorData` element indices relative to
    /// the shape. Absolute indices are TensorView.viewOffset + these values
    var relativeIndices: DataShapeSequence {
        return DataShapeSequence(shape: self, dimension: 0, startingIndex: 0)
    }
}

//==============================================================================
// NDIndexSequence
//extension DataShape {
//    var
//}



////==============================================================================
//// DataShape
//extension DataShape: Sequence {
//    public func makeIterator() -> NDExtentIterator {
//        return NDExtentIterator(shape: self, extentIndex: 0,
//                                startingIndex: 0)
//    }
//}

////==============================================================================
//// DataShapeIterator
//// This is a recursive iterator that works it's way through N dimensions
//public protocol DataShapeIterator: Sequence, IteratorProtocol {
//    /// the index of the extent being iterated
//    var _extentIndex: Int { get set }
//    /// the current iterator tensor index
//    var _currentTensorIndex: Int { get set }
//    /// the tensorIndex just past the end of this range
//    var _tensorIndexPastEnd: Int { get set }
//    /// the shape being iterated
//    var _shape: DataShape { get set }
//    /// returns an iterator for the tensor data indices for the elements
//    /// contained by this extent, which is the next dimension
//    var tensorIndices: IndexSequence { get }
//
//    /// initializer
//    /// - Parameter shape: the shape being iterated
//    /// - Parameter extentIndex: the extent being iterated
//    /// - Parameter startingIndex: the starting tensor index
//    init(shape: DataShape, extentIndex: Int, startingIndex: Int)
//
//    /// makes a sequence iterator for the next dimension
//    func makeIterator() -> Self
//
//    /// return iterator for the next dimension
//    mutating func next() -> Self?
//}
//
//public extension DataShapeIterator {
//    /// makes a sequence iterator for the next dimension
//    func makeIterator() -> Self {
//        assert(_extentIndex + 1 < _shape.rank)
//        return Self.init(shape: _shape,
//                         extentIndex: _extentIndex + 1,
//                         startingIndex: _currentTensorIndex)
//    }
//
//    /// return iterator for the next dimension
//    mutating func next() -> Self? {
//        // if we pass the end then return nil
//        if _currentTensorIndex == _tensorIndexPastEnd {
//            return nil
//        } else {
//            let iter = Self.init(shape: _shape,
//                                 extentIndex: _extentIndex,
//                                 startingIndex: _currentTensorIndex)
//            _currentTensorIndex += _shape.strides[_extentIndex]
//            return iter
//        }
//    }
//
//    /// returns an iterator for the tensor data indices for the elements
//    /// contained by this extent, which is the next dimension
//    var tensorIndices: IndexSequence {
//        assert(_extentIndex + 1 < _shape.rank)
//        return IndexSequence(shape: _shape, dim: _extentIndex + 1,
//                             offset: _currentTensorIndex)
//    }
//}
//
////==============================================================================
//// NDExtentIterator
//public struct NDExtentIterator: DataShapeIterator {
//    // properties
//    public var _currentTensorIndex: Int
//    public var _extentIndex: Int
//    public var _shape: DataShape
//    public var _tensorIndexPastEnd: Int
//
//    /// initializer
//    public init(shape: DataShape, extentIndex: Int, startingIndex: Int) {
//        _shape = shape
//        _extentIndex = extentIndex
//        _currentTensorIndex = startingIndex
//        _tensorIndexPastEnd = startingIndex +
//            shape.extents[extentIndex] * shape.strides[extentIndex]
//    }
//
//}
//
////==============================================================================
//// IndexSequence
//public struct IndexSequence: Sequence {
//    let dim: Int
//    var offset: Int
//    let shape: DataShape
//
//    init(shape: DataShape, dim: Int, offset: Int) {
//        self.dim = dim
//        self.offset = offset
//        self.shape = shape
//    }
//
//    public func makeIterator() -> IndexIterator {
//        return IndexIterator(shape: shape, dim: dim, offset: offset)
//    }
//}
//
//
//public struct IndexIterator: IteratorProtocol {
//    let dim: Int
//    let endOffset: Int
//    var offset: Int
//    let shape: DataShape
//
//    init(shape: DataShape, dim: Int, offset: Int) {
//        self.dim = dim
//        self.endOffset = offset + shape.extents[dim] * shape.strides[dim]
//        self.offset = offset
//        self.shape = shape
//    }
//
//    public mutating func next() -> Int? {
//        // if we pass the end then return nil
//        if offset == endOffset {
//            return nil
//        } else {
//            let value = offset
//            offset += shape.strides[dim]
//            return value
//        }
//    }
//}
//
