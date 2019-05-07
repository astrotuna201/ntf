//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndex {
    // properties
    public var viewIndex: Int = 0
    public var dataIndex: Int = 0
    public var isPad: Bool = false
    
    // local properties
    public let rowBounds: ExtentBounds
    public let colBounds: ExtentBounds
    public var row: Int
    public var col: Int
    public let isPadded: Bool
    public let isRepeated: Bool
    
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        let bounds = view.createTensorBounds()
        rowBounds = bounds[0]
        colBounds = bounds[1]
        isPadded = view.isPadded
        isRepeated = view.isRepeated
        row = position.r
        col = position.c

        computeIndexes()
    }
    
    @inlinable @inline(__always)
    public mutating func computeIndexes() {
        viewIndex = row * rowBounds.viewStride + col * colBounds.viewStride
        
        func indexFrom(_ r: Int, _ c: Int) -> Int {
            return r * rowBounds.dataStride + c * colBounds.dataStride
        }

        func repeatedFrom(_ r: Int, _ c: Int) -> Int {
            return (r % rowBounds.viewExtent) * rowBounds.dataStride +
                (c % colBounds.viewExtent) * colBounds.dataStride
        }

        if isPadded {
            isPad =
                row < rowBounds.before || row >= rowBounds.after ||
                col < colBounds.before || col >= colBounds.after
            
            if !isPad {
                let r = row - rowBounds.before
                let c = col - colBounds.before
                dataIndex = isRepeated ? repeatedFrom(r, c) : indexFrom(r, c)
            }
        } else if isRepeated {
            dataIndex = repeatedFrom(row, col)
        } else {
            dataIndex = indexFrom(row, col)
        }
    }

    
    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        next.col += 1
        
        if isPadded {
            next.isPad =
                next.row < rowBounds.before || next.row >= rowBounds.after ||
                next.col < colBounds.before || next.col >= colBounds.after
            if next.isPad { return next }
        }

        if isRepeated {
            if next.col < colBounds.viewExtent {
                next.dataIndex = (next.col % colBounds.dataExtent) * colBounds.dataStride
            } else {
                next.col = 0
                next.row += 1
                next.dataIndex = (next.row % rowBounds.dataExtent) * rowBounds.dataStride
            }
        } else {
            if next.col < colBounds.viewExtent {
                next.dataIndex = dataIndex + colBounds.dataStride
            } else {
                next.col = 0
                next.row += 1
                next.dataIndex = next.row * rowBounds.dataStride
            }
        }
        next.viewIndex += 1
        return next
    }

    @inlinable @inline(__always)
    public func advanced(by n: Int) -> MatrixIndex {
        guard n != 1 else { return increment() }

        // update the row and column positions
        let jump = n.quotientAndRemainder(dividingBy: rowBounds.viewExtent)
        var next = self
        next.row += jump.quotient
        next.col += jump.remainder
        
        // now set the indexes
        next.computeIndexes()
        return next
    }
}
