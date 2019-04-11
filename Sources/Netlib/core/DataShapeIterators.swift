//******************************************************************************
//  Created by Edward Connell on 4/5/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// ExtentPosition
/// This is used to track the iterated position for each extent in the view
public struct ExtentPosition {
    /// the position for the `view` being traversed, which might be
    /// different than the data view and includes padding
    var view: ShapePosition
    /// the position for the real `data` being traveresed.
    var data: ShapePosition
    /// the current position in this dimension is padding
    var currentIsPad: Bool
    /// the current position in this dimension is padding
    var parentIsPad: Bool
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

public struct ShapePosition {
    /// the base offset for this iteration sequence
    //    var base: Int
    /// current cummulative iterative position accross the shapes
    var current: Int
    /// the strided span of the extent
    let span: Int
    /// the position just after the last element
    var end: Int
}

public struct DataShapeIndex {
    let viewPos: Int
    let dataPos: Int
}

//==============================================================================
// DataShapeSequenceIterable
// This is a recursive iterator that works it's way through N dimensions
public protocol DataShapeSequenceIterable: IteratorProtocol {
    /// types
    typealias AdvanceFunction =
        (_ position: inout [ExtentPosition], _ dim: Int) -> DataShapeIndex?

    /// function used to advance the position. This can be for concrete or
    /// virtual shapes.
    var advanceFn: AdvanceFunction! { get set }
    /// the view of the data that will be iterated over
    var data: DataShape { get set }
    /// the relative offset to add to each index
    var offset: Int { get set }
    /// the current position in nD space
    var position: [ExtentPosition] { get set }
    /// the view being iterated
    var view: DataShape { get set }
    /// the void space before and after each dimension
    var padding: [Padding] { get set }
    /// fully specified initializer
    init(view: DataShape, at offset: Int,
         repeating repeatedShape: DataShape?,
         with padding: [Padding]?)
}

// shorthand

//==============================================================================
// DataShapeSequenceIterable default implementation
public extension DataShapeSequenceIterable {
    
    //--------------------------------------------------------------------------
    /// next
    /// advances to the next position in the view.
    /// - Returns: The `tensorData` buffer index associated with this view.
    ///   If the index is within a padded region, then -1 is returned.
    mutating func next() -> Int? {
        /// the advance function is selected at init depending on whether
        /// the view is padded or is repeating another view. If the `view`
        /// and `repeatedShape` are the same, then `shapeIndex` and
        /// `repeatedIndex` are equal. If the shapes are not the same, then
        /// `view` is interpreted as virtual, and the `repeatedShape`
        /// represents a real tensorBuffer, therefore the `repeatedIndex` is
        /// always returned.
        return advanceFn(&position, view.lastDimension)?.dataPos
    }
    
    //==========================================================================
    /// advanceFirst(position:for:
    /// sets up the first position for normal indexing. This is only called
    /// once per sequence iteration.
    /// Initialization moves from outer dimension to inner (0 -> rank)
    /// - Returns: the index of the first position. If the view is empty then
    ///   `nil` is returned
    func advanceFirst(_ position: inout [ExtentPosition]) -> DataShapeIndex? {
        guard !view.isEmpty else { return nil }

        // get the padding and set an increment if there is more than one
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0

        for dim in 0..<view.rank {
            let before = padding[padIndex].before
            let after = padding[padIndex].after
            let extent = view.extents[dim]
            let stride = view.strides[dim]
            
            // the strided span of this dimension
            let span = (before + after + extent) * stride
            let beforeSpan = before * stride
            let afterSpan = span - after * stride

            // set the current position and end
            let current = offset
            let end = current + span

            // if index 0 of any dimension is in the pad area, then all
            // contained dimensions are padding as well
            let parentIsPad = dim > 0 && position[dim - 1].currentIsPad
            let currentIsPad = parentIsPad || before > 0
            padIndex += padIncrement

            // setup the initial position relative to the data view
            let dcurrent = offset
            let dspan = data.extents[dim] * data.strides[dim]
            let dend = dcurrent + dspan

            // append the fully initialized first position
            position.append(ExtentPosition(
                view: ShapePosition(current: current, span: span, end: end),
                data: ShapePosition(current: dcurrent, span: dspan, end: dend),
                currentIsPad: currentIsPad,
                parentIsPad: parentIsPad,
                padBefore: beforeSpan,
                padBeforeSpan: beforeSpan,
                padAfter: afterSpan,
                padAfterSpan: afterSpan))
        }

        // the first index is 0 plus the caller specified view offset
        // this is usually the TensorView.viewOffset value
        let firstIsPad = position[view.lastDimension].currentIsPad
        return DataShapeIndex(viewPos: offset,
                              dataPos: firstIsPad ? -1 : offset)
    }

    //--------------------------------------------------------------------------
    /// advance(position:for:
    /// advances the index in the lastDimension. If it can't, then the parent
    /// dimension is called recursively to advance.
    /// - Returns: the index of the next position
    /// Minimal per value: 2 cmp, 1 inc
    func advance(_ index: inout [ExtentPosition], dim: Int) -> DataShapeIndex? {
        // check for initial position
        guard index.count > 0 else { return advanceFirst(&index) }
        
        // advance the position for this dimension by it's stride
        index[dim].view.current += view.strides[dim]
        
        // if at the end then go back a dimension and advance
        if index[dim].view.current == index[dim].view.end {
            // make a recursive call to the parent dimension
            if dim > 0, let start = advance(&index, dim: dim - 1) {
                // update the cumulative view position
                index[dim].view.current = start.viewPos
                index[dim].view.end = start.viewPos + index[dim].view.span
                return start
            } else {
                return nil
            }
        } else {
            // In this case the viewIndex and dataIndex are the same
            // becasue there is no repeating or padding
            let current = index[dim].view.current
            return DataShapeIndex(viewPos: current, dataPos: current)
        }
    }
    
    //--------------------------------------------------------------------------
    /// advanceRepeated(position:for:
    /// advances the index in the lastDimension. If it can't, then the parent
    /// dimension is called recursively to advance.
    /// - Returns: the index of the next position
    /// Minimal cost per value: 3 cmp, 2 inc
    func advanceRepeated(_ index: inout [ExtentPosition],
                         for dim: Int) -> DataShapeIndex? {
        // check for initial position
        guard index.count > 0 else { return advanceFirst(&index) }
        
        //--------------------------------
        // advance the `data` position for this dimension by it's stride
        index[dim].data.current += data.strides[dim]
        
        // if past the data end, then go back to beginning and repeat
        if index[dim].data.current == index[dim].data.end {
            index[dim].data.current -= index[dim].data.span
        }
        
        //--------------------------------
        // advance the `view` position for this dimension by it's stride
        index[dim].view.current += view.strides[dim]
        
        // if past the end, then advance parent dimension
        if index[dim].view.current == index[dim].view.end {
            // make a recursive call to advance the parent dimension
            if dim > 0, let start = advanceRepeated(&index, for: dim - 1) {
                // update the cumulative view position
                let current = start.viewPos
                index[dim].view.current = current
                index[dim].view.end = current + index[dim].view.span
                
                // update the cumulative repeated view position
                index[dim].data.current = start.dataPos
                index[dim].data.end = start.dataPos + index[dim].data.span
                return start
            } else {
                // can't advance any further, so return nil
                return nil
            }
        } else {
            // we are not at the end, so return the current position
            return DataShapeIndex(
                viewPos: index[dim].view.current,
                dataPos: index[dim].data.current)
        }
    }

    //==========================================================================
    /// advancePadded(position:for:
    /// Advances the last dimension. If it can't, then `nil` is returned
    /// This function is called recursively.
    /// - Returns: the index of the next position
    ///
    /// Minimal cost per value: 6 cmp, 1 inc, 1 sub
    func advancePadded(_ index: inout [ExtentPosition],
                       for dim: Int) -> DataShapeIndex? {
        // advance to first if needed
        guard index.count > 0 else { return advanceFirst(&index) }

        //--------------------------------
        // advance the `data` position for this dimension by it's stride
        index[dim].data.current += data.strides[dim]
        
        // if past the data end, then go back to beginning and repeat
        if index[dim].data.current == index[dim].data.end {
            index[dim].data.current -= index[dim].data.span
        }
        
        //--------------------------------
        // advance the `view` position for this dimension by it's stride
        index[dim].view.current += view.strides[dim]

        // if at the end then go back a dimension and advance
        if index[dim].view.current == index[dim].view.end {

            // make a recursive call to the parent dimension
            // `start` is the first position in the parent dimension
            if dim > 0, let start = advancePadded(&index, for: dim - 1) {
                // update the cumulative view position and set the new end
                let current = start.viewPos
                index[dim].view.current = current
                index[dim].view.end = current + index[dim].view.span
                
                // update the cumulative repeated view position
                index[dim].data.current = start.dataPos
                index[dim].data.end =
                    start.dataPos + index[dim].data.span
                
                // if the enclosing parent dimension for this is padding
                // then all of this extent is padding
                if index[dim - 1].currentIsPad {
                    index[dim].currentIsPad = true
                } else {
                    // in the case that the parent dimension is not padding,
                    // set the boundaries testing
                    let beforeSpan = index[dim].padBeforeSpan
                    let afterSpan = index[dim].padAfterSpan
                    index[dim].padBefore = current + beforeSpan
                    index[dim].padAfter = current + afterSpan
                }
            }
        }
        
        // the current index is passed on to continue view traversal
        let current = index[dim].view.current
        
        let isPad =
            current < index[dim].padBefore ||
            current >= index[dim].padAfter ||
            (dim > 0 && index[dim - 1].currentIsPad)
        
        index[dim].currentIsPad = isPad
        let repeatedIndex = isPad ? -1 : index[dim].data.current

        // return new position
        assert(repeatedIndex < data.elementSpanCount)
        return DataShapeIndex(viewPos: current,
                              dataPos: repeatedIndex)
    }
}

//==============================================================================
/// DataShapeSequenceIterator
/// This iterates the tensorData indexes described by
/// an N dimensional DataShape as a single linear Sequence
public struct DataShapeSequenceIterator: DataShapeSequenceIterable {
    // properties
    public var advanceFn: AdvanceFunction!
    public var offset: Int
    public var position = [ExtentPosition]()
    public var data: DataShape
    public var view: DataShape
    public var padding: [Padding]

    // initializers
    public init(view: DataShape, at offset: Int,
                repeating dataShape: DataShape?,
                with padding: [Padding]?) {
        self.data = dataShape ?? view
        self.view = view
        self.offset = offset
        self.padding = padding ?? [Padding(before: 0, after: 0)]
        
        if padding != nil {
            advanceFn = advancePadded(_:for:)
        } else if dataShape != nil {
            advanceFn = advanceRepeated(_:for:)
        } else {
            advanceFn = advance(_: dim:)
        }
    }
}

//==============================================================================
// DataShapeSequence
public struct DataShapeSequence: Sequence {
    // properties
    let offset: Int
    let view: DataShape
    let repeatedShape: DataShape?
    let padding: [Padding]?
    
    // initializers
    public init(view: DataShape, at offset: Int,
                repeating repeatedShape: DataShape?,
                with padding: [Padding]?) {
        self.view = view
        self.offset = offset
        self.repeatedShape = repeatedShape
        self.padding = padding
    }
    
    // makeIterator
    public func makeIterator() -> DataShapeSequenceIterator {
        return DataShapeSequenceIterator(view: view, at: offset,
                                         repeating: repeatedShape,
                                         with: padding)
    }
}

extension DataShape {
    /// returns a Sequence of `tensorData` element indices relative to
    /// the specified offset
    func indices(repeating view: DataShape? = nil,
                 relativeTo offset: Int = 0,
                 with padding: [Padding]? = nil) -> DataShapeSequence {
        return DataShapeSequence(view: self, at: offset,
                                 repeating: view, with: padding)
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
//        return NDExtentIterator(view: self, extentIndex: 0,
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
//    /// the view being iterated
//    var _shape: DataShape { get set }
//    /// returns an iterator for the tensor data indices for the elements
//    /// contained by this extent, which is the next dimension
//    var tensorIndices: IndexSequence { get }
//
//    /// initializer
//    /// - Parameter view: the view being iterated
//    /// - Parameter extentIndex: the extent being iterated
//    /// - Parameter startingIndex: the starting tensor index
//    init(view: DataShape, extentIndex: Int, startingIndex: Int)
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
//        return Self.init(view: _shape,
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
//            let iter = Self.init(view: _shape,
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
//        return IndexSequence(view: _shape, dim: _extentIndex + 1,
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
//    public init(view: DataShape, extentIndex: Int, startingIndex: Int) {
//        _shape = view
//        _extentIndex = extentIndex
//        _currentTensorIndex = startingIndex
//        _tensorIndexPastEnd = startingIndex +
//            view.extents[extentIndex] * view.strides[extentIndex]
//    }
//
//}
//
////==============================================================================
//// IndexSequence
//public struct IndexSequence: Sequence {
//    let dim: Int
//    var offset: Int
//    let view: DataShape
//
//    init(view: DataShape, dim: Int, offset: Int) {
//        self.dim = dim
//        self.offset = offset
//        self.view = view
//    }
//
//    public func makeIterator() -> IndexIterator {
//        return IndexIterator(view: view, dim: dim, offset: offset)
//    }
//}
//
//
//public struct IndexIterator: IteratorProtocol {
//    let dim: Int
//    let endOffset: Int
//    var offset: Int
//    let view: DataShape
//
//    init(view: DataShape, dim: Int, offset: Int) {
//        self.dim = dim
//        self.endOffset = offset + view.extents[dim] * view.strides[dim]
//        self.offset = offset
//        self.view = view
//    }
//
//    public mutating func next() -> Int? {
//        // if we pass the end then return nil
//        if offset == endOffset {
//            return nil
//        } else {
//            let value = offset
//            offset += view.strides[dim]
//            return value
//        }
//    }
//}
//
