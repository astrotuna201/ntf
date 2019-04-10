//******************************************************************************
//  Created by Edward Connell on 4/5/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// DataShapeExtentPosition
/// This is used to track the iterated position for each extent in the shape
public struct ShapePosition {
    /// current cummulative iterative position accross the shapes
    var current: Int
    /// the span of the shapes extent including stride
    let span: Int
    /// the position just after the end of the extent in the shapes
    var end: Int
}

public struct DataShapeExtentPosition {
    /// the position for the `shape`
    var shape: ShapePosition
    /// the position for the `repeatedShape`
    var repeated: ShapePosition
    /// All positions before this are padding.
    /// An index of -1 is returned for each padded position
    var padBefore: Int
    /// the relative span size for the before padding
    let padBeforeSpan: Int
    /// All positions after this are padding.
    /// An index of -1 is returned for each padded position
    var padAfter: Int
    /// the relative span size for the after padding
    let padAfterSpan: Int
}

public typealias DataShapeAdvanceIndex = (shapeIndex: Int, repeatedIndex: Int)

//==============================================================================
// DataShapeSequenceIterable
// This is a recursive iterator that works it's way through N dimensions
public protocol DataShapeSequenceIterable: IteratorProtocol {
    /// function used to advance the position. This can be for concrete or
    /// virtual shapes.
    var advanceFn: DataShapeAdvanceFn! { get set }
    /// the shape of the data that will be repeated to support broadcasting
    var repeatedShape: DataShape { get set }
    /// the relative offset to add to each index
    var offset: Int { get set }
    /// the current position in nD space
    var position: [DataShapeExtentPosition]? { get set }
    /// the shape being iterated
    var shape: DataShape { get set }
    /// the void space before and after each dimension
    var padding: [Padding]? { get set }
    /// fully specified initializer
    init(shape: DataShape, at offset: Int,
         repeating repeatedShape: DataShape?,
         with padding: [Padding]?)
}

// shorthand
public typealias DataShapeAdvanceFn =
    (_ position: inout [DataShapeExtentPosition]?, _ dim: Int) ->
    DataShapeAdvanceIndex?

//==============================================================================
// DataShapeSequenceIterable default implementation
public extension DataShapeSequenceIterable {
    
    //--------------------------------------------------------------------------
    /// next
    /// advances to the next position in the shape.
    /// - Returns: The `tensorData` buffer index associated with this shape.
    ///   If the index is within a padded region, then -1 is returned.
    mutating func next() -> Int? {
        /// the advance function is selected at init depending on whether
        /// the shape is padded or is repeating another shape. If the `shape`
        /// and `repeatedShape` are the same, then `shapeIndex` and
        /// `repeatedIndex` are equal. If the shapes are not the same, then
        /// `shape` is interpreted as virtual, and the `repeatedShape`
        /// represents a real tensorBuffer, therefore the `repeatedIndex` is
        /// always returned.
        return advanceFn(&position, shape.lastDimension)?.repeatedIndex
    }
    
    //==========================================================================
    /// advanceFirst(position:for:
    /// sets up the first position for normal indexing. This is only called
    /// once per sequence iteration.
    /// - Returns: the index of the first position. If the shape is empty then
    ///   `nil` is returned
    func advanceFirst(_ position: inout [DataShapeExtentPosition]?,
                      for dim: Int) -> DataShapeAdvanceIndex? {
        guard !shape.isEmpty else { return nil }

        // get the padding and set an increment if there is more than one
        let padding = self.padding ?? [Padding(before: 0, after: 0)]
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0

        // record the starting point for each dimension
        position = [DataShapeExtentPosition]()
        
        // if index 0 of any dimension is in the pad area, then all
        // contained dimensions are padding as well
        var firstIndexIsPad = padding[0].before > 0

        for dim in 0..<shape.rank {
            // the strided span of this dimension
            let beforeSize = padding[padIndex].before
            let afterSize = padding[padIndex].after
            let span = (beforeSize + afterSize + shape.extents[dim]) *
                shape.strides[dim]
            
            // set the current position and end
            let current = offset
            let end = current + span

            // setting before equal to the dimension's span makes it all pad
            let beforeSpan = firstIndexIsPad ? span :
                beforeSize * shape.strides[dim]

            // the after boundary will be tested assuming it is relative to
            // the start, so subtract it from the end
            let afterSpan = firstIndexIsPad ? span :
                span - afterSize * shape.strides[dim]

            // advance the padding index for the multi pad case
            padIndex += padIncrement

            // if before is greater than 0 for this dimension, then subsequent
            // dimensions are also pad
            if beforeSize > 0 { firstIndexIsPad = true }
            
            // setup the initial position relative to the repeated shape
            let rspan = repeatedShape.extents[dim] *
                repeatedShape.strides[dim]
            let rcurrent = offset
            let rend = rcurrent + rspan
            let rpos = ShapePosition(current: rcurrent, span: rspan, end: rend)

            // append the fully initialized first position
            position!.append(DataShapeExtentPosition(
                shape: ShapePosition(current: current, span: span, end: end),
                repeated: rpos,
                padBefore: beforeSpan,
                padBeforeSpan: beforeSpan,
                padAfter: afterSpan,
                padAfterSpan: afterSpan))
        }

        // the first index is 0 plus the caller specified shape offset
        // this is usually the TensorView.viewOffset value
        return (shapeIndex: offset, repeatedIndex: offset)
    }

    //--------------------------------------------------------------------------
    /// advancePosition(position:for:
    /// Advances the last dimension. If it can't, then `nil` is returned
    /// This function is called recursively.
    /// - Returns: the index of the next position
    func advance(_ position: inout [DataShapeExtentPosition]?,
                 for dim: Int) -> DataShapeAdvanceIndex? {
        // check for initial position
        var nextPos: DataShapeAdvanceIndex?
        guard position != nil else { return advanceFirst(&position, for: dim) }
        
        // advance the position for this dimension by it's stride
        position![dim].shape.current += shape.strides[dim]
        
        // if at the end then go back a dimension and advance
        if position![dim].shape.current == position![dim].shape.end {
            // make a recursive call to the parent dimension
            if dim > 0, let start = advance(&position, for: dim - 1) {
                // update the cumulative shape position
                let current = start.shapeIndex
                position![dim].shape.current = current
                position![dim].shape.end = current + position![dim].shape.span
                nextPos = start
            }
        } else {
            // this function does not account for repeating or padding,
            // so the current position is used for both
            let current = position![dim].shape.current
            nextPos = (shapeIndex: current, repeatedIndex: current)
        }

        return nextPos
    }
    
    //==========================================================================
    /// getRepeatedIndex helper
    private func padTest(_ position: DataShapeExtentPosition) -> Int {
        // check if current falls within padding area
        let current = position.shape.current
        let repeatedIndex = position.padBefore..<position.padAfter ~= current ?
            current - position.padBefore : -1
        return repeatedIndex
    }
    
    //==========================================================================
    /// advancePosition(position:for:
    /// Advances the last dimension. If it can't, then `nil` is returned
    /// This function is called recursively.
    /// - Returns: the index of the next position
    func advancePadded(_ position: inout [DataShapeExtentPosition]?,
                       for dim: Int) -> DataShapeAdvanceIndex? {
        // advance to the first position if it is `nil`
        var nextPos: DataShapeAdvanceIndex?
        if position == nil {
            _ = advanceFirst(&position, for: dim)
            nextPos = (position![dim].shape.current, padTest(position![dim]))
            return nextPos
        }

        // advance the position for this dimension by it's stride
        position![dim].shape.current += shape.strides[dim]
        
        // if at the end then go back a dimension and advance
        if position![dim].shape.current == position![dim].shape.end {

            // make a recursive call to the parent dimension
            // `start` is the first position in the parent dimension
            if dim > 0, let start = advancePadded(&position, for: dim - 1) {
                // update the cumulative shape position and set the new end
                let current = start.shapeIndex
                position![dim].shape.current = current
                position![dim].shape.end = current + position![dim].shape.span
                
                // a repeated index of -1 signifies an index into padding
                if start.repeatedIndex == -1 {
                    // the enclosing parent dimension for this is padding
                    // so all of this is padding.
                    let shapeSpan = position![dim].shape.span
                    position![dim].padBefore = current + shapeSpan
                    position![dim].padAfter = current + shapeSpan
                } else {
                    // in the case that the parent dimension is not padding,
                    // set the boundaries testing
                    let beforeSpan = position![dim].padBeforeSpan
                    let afterSpan = position![dim].padAfterSpan
                    position![dim].padBefore = current + beforeSpan
                    position![dim].padAfter = current + afterSpan
                }
                // return the position
                nextPos = start
            }
        } else {
            // the current index is passed on to continue shape traversal
            // the current index is tested to see if it falls into the
            // padding region or not
            nextPos = (shapeIndex: position![dim].shape.current,
                       repeatedIndex: padTest(position![dim]))
        }
        
        return nextPos
    }
    
    //--------------------------------------------------------------------------
    /// advanceRepeated(position:for:
    /// advances the lastDimension . If it can't, then `position`
    /// is set to `nil` this is a recursive function
    /// - Returns: the index of the next position
    func advanceRepeated(_ position: inout [DataShapeExtentPosition]?,
                         for dim: Int) -> DataShapeAdvanceIndex? {
        // check for initial position
        var nextPos: DataShapeAdvanceIndex?
        guard position != nil else { return advanceFirst(&position, for: dim) }

        //--------------------------------
        // advance the `repeatedShape` position for this dimension by stride
        position![dim].repeated.current += repeatedShape.strides[dim]
        
        // if past the end of the repeated dimension, go back to the beginning
        if position![dim].repeated.current == position![dim].repeated.end {
            position![dim].repeated.current -= position![dim].repeated.span
        }

        //--------------------------------
        // advance the `shape` position for this dimension by stride
        position![dim].shape.current += shape.strides[dim]
        
        // if past the end of this dimension,
        // then go back a dimension and advance
        if position![dim].shape.current == position![dim].shape.end {
            // make a recursive call to the parent dimension
            if dim > 0, let start = advanceRepeated(&position, for: dim - 1) {
                // update the cumulative shape position
                let current = start.shapeIndex
                position![dim].shape.current = current
                position![dim].shape.end = current + position![dim].shape.span

                // update the cumulative repeated shape position
                position![dim].repeated.current = start.repeatedIndex
                position![dim].repeated.end =
                    start.repeatedIndex + position![dim].repeated.span

                // return the next position
                nextPos = start
            }
        } else {
            nextPos = (position![dim].shape.current,
                       position![dim].repeated.current)
        }
        
        return nextPos
    }
}

//==============================================================================
/// DataShapeSequenceIterator
/// This iterates the tensorData indexes described by
/// an N dimensional DataShape as a single linear Sequence
public struct DataShapeSequenceIterator: DataShapeSequenceIterable {
    // properties
    public var advanceFn: DataShapeAdvanceFn!
    public var offset: Int
    public var position: [DataShapeExtentPosition]?
    public var repeatedShape: DataShape
    public var shape: DataShape
    public var padding: [Padding]?

    // initializers
    public init(shape: DataShape, at offset: Int,
                repeating repeatShape: DataShape?,
                with padding: [Padding]?) {
        self.repeatedShape = repeatShape ?? shape
        self.shape = shape
        self.offset = offset
        self.padding = padding
        
        if padding != nil {
            if repeatShape == nil {
                advanceFn = advancePadded(_:for:)
            } else {
                fatalError("not implemented")
            }
        } else if repeatShape == nil {
            advanceFn = advance(_:for:)
        } else {
            advanceFn = advanceRepeated(_:for:)
        }
    }
}

//==============================================================================
// DataShapeSequence
public struct DataShapeSequence: Sequence {
    // properties
    let offset: Int
    let shape: DataShape
    let repeatedShape: DataShape?
    let padding: [Padding]?
    
    // initializers
    public init(shape: DataShape, at offset: Int,
                repeating repeatedShape: DataShape?,
                with padding: [Padding]?) {
        self.shape = shape
        self.offset = offset
        self.repeatedShape = repeatedShape
        self.padding = padding
    }
    
    // makeIterator
    public func makeIterator() -> DataShapeSequenceIterator {
        return DataShapeSequenceIterator(shape: shape, at: offset,
                                         repeating: repeatedShape,
                                         with: padding)
    }
}

extension DataShape {
    /// returns a Sequence of `tensorData` element indices relative to
    /// the specified offset
    func indices(repeating shape: DataShape? = nil,
                 relativeTo offset: Int = 0,
                 with padding: [Padding]? = nil) -> DataShapeSequence {
        return DataShapeSequence(shape: self, at: offset,
                                 repeating: shape, with: padding)
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
