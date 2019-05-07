//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndex {
    // properties
    public let advanceFn: AdvanceFn
    public var viewIndex: Int
    public var dataIndex: Int
    public var isPad: Bool
    public var row: Int
    public var col: Int
    public let bounds: ContiguousArray<ExtentBounds>
    public let isPadded = false
    public let isRepeated = false
    public init(r: Int, c: Int,
                fn: @escaping AdvanceFn,
                view: Int, data: Int, pad: Bool) {
        row = r
        col = c
        advanceFn = fn
        viewIndex = view
        dataIndex = data
        isPad = pad
        
        let b = ExtentBounds(before: 0, after: 0, viewExtent: 10,
                             viewStride: 1, dataExtent: 1, dataStride: 1)
        bounds = ContiguousArray<ExtentBounds>(repeating: b, count: 2)
    }

    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        if isPadded {
            
        } else if isRepeated {
            
        } else {
            // most frequent increment
            next.col = col + 1
            if next.col < bounds[0].viewExtent {
                next.row = row
                next.dataIndex = dataIndex + bounds[0].dataStride
            } else {
                next.row = row + 1
                next.col = 0
                next.dataIndex = next.row * bounds[1].dataStride
            }
        }
        return next
    }

    @inlinable @inline(__always)
    public func advance(by n: Int) -> MatrixIndex {
        guard n != 1 else { return increment() }
        var next = self
        if isPadded {
            
        } else if isRepeated {
            
        } else {
            // incremental jump
            let jump = n.quotientAndRemainder(dividingBy: bounds[1].viewExtent)
            next.row = row + jump.quotient
            next.col = col + jump.remainder
            next.dataIndex = next.row * bounds[1].dataStride + next.col * bounds[0].dataStride
        }
        return next
    }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    var endIndex: MatrixIndex {
        return createIndex(at: (shape.extents[0], 0))
    }

    var startIndex: MatrixIndex {
        return createIndex(at: (0, 0))
    }
}

public extension MatrixView {
    //--------------------------------------------------------------------------
    /// createIndex
    /// The advance functions are used by collections that are bounds
    /// prechecked, so only valid positions are requested
    func createIndex(at position: MatrixPosition) -> MatrixIndex {
        let rows = shape.extents[0]
        let cols = shape.extents[1]
        let dataRows = dataShape.extents[0]
        let dataCols = dataShape.extents[1]
        let rowStride = dataShape.strides[0]
        let colStride = dataShape.strides[1]
        
        // the view position is the same for all situations
        let viewIndex = position.r * shape.strides[0] + position.c
        
        // set the initial position
        let dataIndex: Int
        let isPad: Bool
        let advanceFn: MatrixIndex.AdvanceFn
        
        if isPadded {
            //------------------------------------------------------------------
            fatalError()
        } else if isRepeated {
            //------------------------------------------------------------------
            isPad = false
            dataIndex =
                (position.r % dataRows * rowStride) +
                (position.c % dataCols * colStride)
            
            advanceFn = { i, n in
                var row, col, dataIndex: Int
                // most frequent increment
                if n == 1 {
                    col = i.col + 1
                    if col < cols {
                        row = i.row
                        dataIndex = i.dataIndex + colStride
                    } else {
                        row = i.row + 1
                        col = 0
                        dataIndex = row * rowStride
                    }
                } else {
                    // incremental jump
                    let jump = n.quotientAndRemainder(dividingBy: rows)
                    row = i.row + jump.quotient
                    col = i.col + jump.remainder
                    dataIndex = row * rowStride + col * colStride
                }
                
                return MatrixIndex(r: row, c: col, fn: i.advanceFn,
                                   view: i.viewIndex + n,
                                   data: dataIndex,
                                   pad: false)
            }
        } else {
            //------------------------------------------------------------------
            dataIndex = position.r * dataShape.strides[0] + position.c
            isPad = false

            advanceFn = { i, n in
                var next = i
                // most frequent increment
                if n == 1 {
                    next.col = i.col + 1
                    if next.col < cols {
                        next.row = i.row
                        next.dataIndex = i.dataIndex + colStride
                    } else {
                        next.row = i.row + 1
                        next.col = 0
                        next.dataIndex = next.row * rowStride
                    }
                } else {
                    // incremental jump
                    let jump = n.quotientAndRemainder(dividingBy: rows)
                    next.row = i.row + jump.quotient
                    next.col = i.col + jump.remainder
                    next.dataIndex = next.row * rowStride + next.col * colStride
                }
                return next
            }
        }
        
        // return initial position
        return MatrixIndex(r: position.r, c: position.c, fn: advanceFn,
                           view: viewIndex, data: dataIndex, pad: isPad)
    }
}
