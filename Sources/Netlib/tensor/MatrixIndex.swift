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
    public let traversal: TensorTraversal
    public let rowBounds: ExtentBounds
    public let colBounds: ExtentBounds
    public var row: Int
    public var col: Int
    public let isPadded: Bool
    public let isRepeated: Bool
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        let bounds = view.createTensorBounds()
        rowBounds = bounds[0]
        colBounds = bounds[1]
        row = position.r
        col = position.c

        traversal = initTraversal(nil, false)
        isPadded = view.isPadded
        isRepeated = view.isRepeated
        computeIndexes()
    }

    public init<T>(endOf view: T) where T: TensorView {
        let bounds = view.createTensorBounds()
        rowBounds = bounds[0]
        colBounds = bounds[1]
        row = 0
        col = 0

        traversal = initTraversal(nil, false)
        isPadded = view.isPadded
        isRepeated = view.isRepeated
        viewIndex = view.shape.padded(with: view.padding).elementCount
    }
    
    //--------------------------------------------------------------------------
    /// computeIndexes
    /// direct computation of indexes from position. non incremental.
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

    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        
        func incDataIndex(to c: Int) -> Int {
            switch traversal {
            case .normal:
                return dataIndex + colBounds.dataStride
                
            case .padded:
                return dataIndex + colBounds.dataStride

            case .repeated:
                return (c % colBounds.dataExtent) * colBounds.dataStride
                
            case .paddedRepeated:
                return dataIndex + colBounds.dataStride
            }
        }
        
        func setDataIndex(to r: Int) -> Int {
            switch traversal {
            case .normal:
                return r * rowBounds.dataStride
                
            case .padded:
                return r * rowBounds.dataStride
                
            case .repeated:
                return (r % rowBounds.dataExtent) * rowBounds.dataStride
                
            case .paddedRepeated:
                return (r % rowBounds.dataExtent) * rowBounds.dataStride
            }
        }

        // increment position
        next.viewIndex += 1
        next.col += 1
        
        if next.col < colBounds.viewExtent {
            next.dataIndex = incDataIndex(to: next.col)
        } else {
            next.col = 0
            next.row += 1
            next.dataIndex = setDataIndex(to: next.row)
        }
        
        return next
    }

    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
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
