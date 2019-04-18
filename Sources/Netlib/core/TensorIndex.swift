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
public struct TensorViewCollection<View>: Collection
where View: TensorView {
    // types
    public typealias Scalar = View.Scalar

    // properties
    private let view: View
    private let buffer: UnsafeBufferPointer<Scalar>
    public var startIndex: TensorIndex<View>
    public var endIndex: TensorIndex<View>
    public var count: Int { return view.shape.elementCount }

    
    public init(view: View) throws {
        self.view = view
        buffer = try view.readOnly()
        startIndex = TensorIndex(view, startOffset: view.viewOffset)
        endIndex = TensorIndex(view, endOffset:
            view.viewOffset + view.shape.elementSpanCount)
    }

    //--------------------------------------------------------------------------
    // Collection
    public func index(after i: TensorIndex<View>) -> TensorIndex<View> {
        return i.next()
    }

    public subscript(index: TensorIndex<View>) -> Scalar {
        return index.dataIndex < 0 ? view.padValue : buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorViewCollection
/// returns a readonly collection view of the underlying tensorData.
public struct TensorViewMutableCollection<View>: MutableCollection
where View: TensorView {
    // types
    public typealias Scalar = View.Scalar
    // properties
    private var view: View
    private let buffer: UnsafeMutableBufferPointer<Scalar>
    public var startIndex: TensorIndex<View>
    public var endIndex: TensorIndex<View>
    public var count: Int { return view.shape.elementCount }
    
    
    public init(view: View) throws {
        self.view = view
        buffer = try self.view.readWrite()
        startIndex = TensorIndex(view, startOffset: view.viewOffset)
        endIndex = TensorIndex(view, endOffset:
            view.viewOffset + view.shape.elementSpanCount)
    }

    //--------------------------------------------------------------------------
    // MutableCollectionCollection
    public func index(after i: TensorIndex<View>) -> TensorIndex<View> {
        return i.next()
    }
    
    public subscript(index: TensorIndex<View>) -> Scalar {
        get {
            return index.dataIndex < 0 ? view.padValue : buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}

//==============================================================================
/// ShapePosition
public struct ShapePosition {
    /// current cummulative iterative position accross the shapes
    var current: Int
    /// the strided span of the extent
    let span: Int
    /// the position just after the last element
    var end: Int
}

/// This is used to track the iterated position for each dimension in the view
public struct ExtentPosition {
    /// the position for the `view` being traversed, which might be
    /// different than the data view and includes padding
    var view: ShapePosition
    /// the position for the real `data` being traveresed.
    var data: ShapePosition
    /// the current position in this dimension is padding
    var currentIsPad: Bool
    /// the parent dimension is padding
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

public struct DataShapeIndex {
    /// linear view index
    let viewPos: Int
    /// linear data index
    let dataPos: Int
}

//==============================================================================
/// TensorIndex
public struct TensorIndex<View> : Comparable where View: TensorView {
    // properties
    let view: View
    var currentPosition: Int
    var position = [ExtentPosition]()
    
    var rank: Int { return view.rank }
    var lastDimension: Int { return view.shape.lastDimension }
    var dataIndex: Int { return position[lastDimension].data.current }
    
    // initializers
    public init(_ view: View, startOffset: Int) {
        self.view = view
        currentPosition = startOffset
        initializePosition(at: startOffset)
    }

    public init(_ view: View, endOffset: Int) {
        self.view = view
        currentPosition = endOffset
    }
    
    // Equatable
    public static func == (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.currentPosition == rhs.currentPosition
    }
    
    // Comparable
    public static func < (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.currentPosition < rhs.currentPosition
    }
    
    public func next() -> TensorIndex {
        var index = self
        _ = index.advance(dim: lastDimension)
        return index
    }
    
    //==========================================================================
    /// initializePosition(at offset:
    ///
    /// sets up the first position for indexing. This is only called
    /// once per sequence iteration.
    /// Initialization moves from outer dimension to inner (0 -> rank)
    mutating func initializePosition(at offset: Int) {
        assert(view.shape.elementCount > 0)
        
        // get the padding and set an increment if there is more than one
        let padding = view.padding
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0
        let dataShape = view.dataShape
        
        for dim in 0..<view.rank {
            let before = padding[padIndex].before
            let after = padding[padIndex].after
            let extent = view.shape.extents[dim]
            let stride = view.shape.strides[dim]
            
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
            let dspan = dataShape.extents[dim] * dataShape.strides[dim]
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
    }
    
    //==========================================================================
    /// advance(dim:
    /// Advances the current position
    /// Minimal cost per value: 6 cmp, 1 inc, 1 sub
    private mutating func advance(dim: Int) -> DataShapeIndex {
        //--------------------------------
        // advance the `view` position for this dimension by it's stride
        position[dim].view.current += view.shape.strides[dim]

        //--------------------------------
        // if view position is past the end
        // then advance parent dimension if there is one
        if position[dim].view.current == position[dim].view.end && dim > 0 {
            // make a recursive call to advance the parent dimension
            let start = advance(dim: dim - 1)
            position[dim].parentIsPad = position[dim - 1].currentIsPad
            
            // update the view position
            let current = start.viewPos
            position[dim].view.current = current
            position[dim].view.end = current + position[dim].view.span
            
            // update the data position
            position[dim].data.current = start.dataPos
            position[dim].data.end = start.dataPos + position[dim].data.span
            
            // update the padding ranges
            position[dim].padBefore = current + position[dim].padBeforeSpan
            position[dim].padAfter = current + position[dim].padAfterSpan

            // if the parent is pad or if current is in a padding area
            position[dim].currentIsPad =
                position[dim].parentIsPad ||
                current < position[dim].padBefore ||
                current >= position[dim].padAfter
        } else {
            //--------------------------------
            // advance the `data` position for this dimension by it's stride
            // if we are not in a padded region
            if !position[dim].currentIsPad {
                // advance data index
                position[dim].data.current += view.dataShape.strides[dim]
                
                print("viewPos: \(position[dim].view.current) dataPos: \(position[dim].data.current)")

                // if past the data end, then go back to beginning and repeat
                if position[dim].data.current == position[dim].data.end {
                    position[dim].data.current -= position[dim].data.span
                }
            }
        }
        // return the new position
        currentPosition = position[dim].view.current
        return DataShapeIndex(viewPos: currentPosition,
                              dataPos: position[dim].data.current)
    }
}
