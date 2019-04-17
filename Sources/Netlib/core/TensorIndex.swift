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
    
    public init(view: View) throws {
        self.view = view
        buffer = try view.readOnly()
    }

    //--------------------------------------------------------------------------
    // Collection
    public var startIndex: TensorIndex<View> {
        return TensorIndex(view, at: view.viewOffset)
    }
    public var endIndex: TensorIndex<View> {
        let offset = view.viewOffset + view.shape.elementSpanCount
        return view.shape.elementCount == 0 ?
            startIndex : TensorIndex(view, at: offset)
    }
    
    public var count: Int { return view.shape.elementCount }

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
    
    public init(view: View) throws {
        self.view = view
        buffer = try self.view.readWrite()
    }
    
    //--------------------------------------------------------------------------
    // MutableCollectionCollection
    public var startIndex: TensorIndex<View> {
        return TensorIndex(view, at: view.viewOffset)
    }
    public var endIndex: TensorIndex<View> {
        let offset = view.viewOffset + view.shape.elementSpanCount
        return view.shape.elementCount == 0 ?
            startIndex : TensorIndex(view, at: offset)
    }
    
    public var count: Int { return view.shape.elementCount }
    
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
    /// current cummulative iterative position accross the shapes
    var current: Int
    /// the strided span of the extent
    let span: Int
    /// the position just after the last element
    var end: Int
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
    var position = [ExtentPosition]()
    
    var rank: Int { return view.rank }
    var lastDimension: Int { return view.shape.lastDimension }
    var viewIndex: Int { return position[lastDimension].view.current }
    var dataIndex: Int { return position[lastDimension].data.current }
    
    // initializers
    public init(_ view: View, at offset: Int) {
        self.view = view
        initializePosition(at: offset)
    }

    // Equatable
    public static func == (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    public static func < (lhs: TensorIndex, rhs: TensorIndex) -> Bool {
        assert(lhs.rank == rhs.rank)
        return lhs.viewIndex < rhs.viewIndex
    }
    
    public func next() -> TensorIndex {
        fatalError()
    }
    
    //==========================================================================
    /// advanceFirst(position:for:
    /// sets up the first position for normal indexing. This is only called
    /// once per sequence iteration.
    /// Initialization moves from outer dimension to inner (0 -> rank)
    /// - Returns: the index of the first position. If the view is empty then
    ///   `nil` is returned
    mutating func initializePosition(at offset: Int) {
        guard view.shape.elementCount > 0 else { return }

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
        // advance the `view` position for this dimension by it's stride
        position[dim].view.current += view.shape.strides[dim]
        
        // if past the end, then advance parent dimension
        if position[dim].view.current == position[dim].view.end {
            // make a recursive call to advance the parent dimension
            if dim > 0 {
                let start = advance(dim: dim - 1)
                // the current is pad if the parent is pad
                position[dim].currentIsPad = position[dim - 1].currentIsPad
                
                // update the view position
                let current = start.viewPos
                position[dim].view.current = current
                position[dim].view.end = current + position[dim].view.span
                
                // update the data position
                position[dim].data.current = start.dataPos
                position[dim].data.end = start.dataPos + position[dim].data.span
                
                if !position[dim].currentIsPad {
                    // update the padding ranges
                    position[dim].padBefore = current + position[dim].padBeforeSpan
                    position[dim].padAfter = current + position[dim].padAfterSpan
                    
                    // if index 0 of any dimension is in the pad area, then all
                    // contained dimensions are padding as well
                    position[dim].currentIsPad =
                        current < position[dim].padBefore ||
                        current >= position[dim].padAfter
                }
                // Fall through here to perform test on update current
            }
        }
        
        //--------------------------------
        // advance the `data` position for this dimension by it's stride
        // only while we are not in the padded region
        var dataIndex: Int
        let parentIsPad = dim > 0 && position[dim - 1].currentIsPad
        let current = position[dim].view.current
        
        if parentIsPad {
            position[dim].currentIsPad = true
            dataIndex = -1
        } else {
            // if index 0 of any dimension is in the pad area, then all
            // contained dimensions are padding as well
            position[dim].currentIsPad =
                current < position[dim].padBefore ||
                current >= position[dim].padAfter
            
            if position[dim].currentIsPad {
                dataIndex = -1
            } else {
                dataIndex = position[dim].data.current
                position[dim].data.current += view.dataShape.strides[dim]
                
                // if past the data end, then go back to beginning and repeat
                if position[dim].data.current == position[dim].data.end {
                    position[dim].data.current -= position[dim].data.span
                }
            }
        }
        
        // we are not at the end, so return the current position
        return DataShapeIndex(viewPos: current, dataPos: dataIndex)
    }

}
