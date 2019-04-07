//******************************************************************************
//  Created by Edward Connell on 4/5/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
//
public struct ExtentPosition {
    let span: Int
    var current: Int
    var end: Int
    var startData: Int
    var endData: Int
}

public typealias DataShapeAdvanceFn =
    (_ position: inout [ExtentPosition]?, _ dim: Int) -> Int?

//==============================================================================
// DataShapeSequenceIterable
// This is a recursive iterator that works it's way through N dimensions
public protocol DataShapeSequenceIterable: IteratorProtocol {
    /// function used to advance the position. This can be for concrete or
    /// virtual shapes.
    var advanceFn: DataShapeAdvanceFn! { get set }
    /// the relative offset to add to each index
    var offset: Int { get set }
    /// the current position in nD space
    var position: [ExtentPosition]? { get set }
    /// the shape being iterated
    var shape: DataShape { get set }
    /// fully specified initializer
    init(shape: DataShape, at offset: Int)
}

public extension DataShapeSequenceIterable {
    //--------------------------------------------------------------------------
    /// next
    /// advances to the next position in the shape
    mutating func next() -> Int? {
        return advanceFn(&position, shape.lastDimension)
    }

    //--------------------------------------------------------------------------
    /// advanceInitial(position:for:
    /// sets up the initial position and advances the lastDimension.
    /// If the shape is empty then `nil` is returned
    /// - Returns: the index of the next position
    func advanceInitial(_ position: inout [ExtentPosition]?,
                        for dim: Int) -> Int? {
        guard !shape.isEmpty else { return nil }
        var initial = [ExtentPosition]()
        
        // record the starting point for each dimension
        for dim in 0..<shape.rank {
            let span = shape.dataExtents[dim] * shape.strides[dim]
            initial.append(ExtentPosition(span: span,
                                          current: offset,
                                          end: span,
                                          startData: offset,
                                          endData: span))
        }
        
        // return the first position
        position = initial
        return 0
    }

    //--------------------------------------------------------------------------
    /// advancePosition(position:for:
    /// advances the lastDimension. If it can't, then `position`
    /// is set to `nil` this is a recursive function
    /// - Returns: the index of the next position
    func advance(_ position: inout [ExtentPosition]?, for dim: Int) -> Int? {
        guard position != nil else { return advanceInitial(&position, for: dim)}
        var nextPos: Int?
        
        // advance the position for this dimension by it's stride
        position![dim].current += shape.strides[dim]
        
        // if past the end then go back a dimension and advance
        if position![dim].current == position![dim].end {
            // make a recursive call to the parent dimension
            if dim > 0, let start = advance(&position, for: dim - 1) {
                position![dim].current = start
                position![dim].end = start + position![dim].span
                nextPos = start
            }
        } else {
            nextPos = position![dim].current
        }

        return nextPos
    }
    
    //--------------------------------------------------------------------------
    /// advanceInitialVirtual(position:for:
    /// sets up the initial position and advances the lastDimension.
    /// If the shape is empty then `nil` is returned
    /// - Returns: the index of the next position
    func advanceInitialVirtual(_ position: inout [ExtentPosition]?,
                               for dim: Int) -> Int? {
        guard !shape.isEmpty else { return nil }
        var initial = [ExtentPosition]()
        
        // record the starting point for each dimension
        for dim in 0..<shape.rank {
            let span = shape.dataExtents[dim] * shape.strides[dim]
            initial.append(ExtentPosition(span: span,
                                          current: offset,
                                          end: span,
                                          startData: offset,
                                          endData: span))
        }
        
        // return the first position
        position = initial
        return 0
    }
    
    //--------------------------------------------------------------------------
    /// advanceVirtual(position:for:
    /// advances the lastDimension. If it can't, then `position`
    /// is set to `nil` this is a recursive function
    /// - Returns: the index of the next position
    func advanceVirtual(_ position: inout [ExtentPosition]?,
                        for dim: Int) -> Int? {
        guard position != nil else {
            return advanceInitialVirtual(&position, for: dim)
        }
        
        // advance the position for this dimension by it's stride
        position![dim].current += shape.strides[dim]
        
        // if past the end then go back a dimension and advance
        if position![dim].current == position![dim].end {
            // make a recursive call to the parent dimension
            if dim > 0, let start = advance(&position, for: dim - 1) {
                position![dim].current = start
                position![dim].end = start + position![dim].span
                return start
            }
        }
        
        return position![dim].current
    }
}

//==============================================================================
/// DataShapeSequenceIterator
/// This iterates the tensorData indexes described by
/// an N dimensional DataShape as a single linear Sequence
public struct DataShapeSequenceIterator: DataShapeSequenceIterable {
    public var advanceFn: DataShapeAdvanceFn!
    public var offset: Int
    public var position: [ExtentPosition]?
    public var shape: DataShape
    
    public init(shape: DataShape, at offset: Int) {
        self.shape = shape
        self.offset = offset
        advanceFn = shape.isVirtual ? advanceVirtual(_:for:) : advance(_:for:)
    }
}

//==============================================================================
// DataShapeSequence
public struct DataShapeSequence: Sequence {
    let shape: DataShape
    let offset: Int
    
    public init(shape: DataShape, at offset: Int) {
        self.shape = shape
        self.offset = offset
    }
    
    public func makeIterator() -> DataShapeSequenceIterator {
        return DataShapeSequenceIterator(shape: shape, at: offset)
    }
}

extension DataShape {
    /// returns a Sequence of `tensorData` element indices relative to
    /// the specified offset
    func indices(relativeTo offset: Int = 0) -> DataShapeSequence {
        return DataShapeSequence(shape: self, at: offset)
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
