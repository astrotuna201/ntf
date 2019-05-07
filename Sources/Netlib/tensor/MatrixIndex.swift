//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndex {
    // properties
    public var viewIndex: Int
    public var dataIndex: Int
    public var isPad: Bool
    
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
        viewIndex = 0
        dataIndex = 0
        isPad = false
        computeIndexes()
    }
    
    @inlinable @inline(__always)
    public mutating func computeIndexes() {
        if isPadded {
            viewIndex = 0
            dataIndex = 0
            
        } else if isRepeated {
            viewIndex = 0
            dataIndex = 0
            
        } else {
            viewIndex = row * rowBounds.viewStride + col * colBounds.viewStride
            dataIndex = row * rowBounds.dataStride + col * colBounds.dataStride
        }
    }
    
    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        if isPadded {

        } else if isRepeated {
        } else {
            next.col += 1
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
