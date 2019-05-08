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
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        let bounds = view.createTensorBounds()
        rowBounds = bounds[0]
        colBounds = bounds[1]
        row = position.r
        col = position.c
        traversal = view.traversal
        viewIndex = row * rowBounds.viewStride + col * colBounds.viewStride
        computeDataIndex()
    }

    public init<T>(endOf view: T) where T: TensorView {
        let bounds = view.createTensorBounds()
        rowBounds = bounds[0]
        colBounds = bounds[1]
        row = 0
        col = 0
        traversal = view.traversal
        viewIndex = view.shape.padded(with: view.padding).elementCount
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable @inline(__always)
    public mutating func computeDataIndex() {
        func getDataIndex(_ r: Int, _ c: Int) -> Int {
            return r * rowBounds.dataStride + c * colBounds.dataStride
        }

        func getRepeatedDataIndex(_ r: Int, _ c: Int) -> Int {
            return (r % rowBounds.dataExtent) * rowBounds.dataStride +
                (c % colBounds.dataExtent) * colBounds.dataStride
        }
        
        func testIsPad() -> Bool {
            return
                row < rowBounds.before || row >= rowBounds.after ||
                col < colBounds.before || col >= colBounds.after
        }

        //----------------------------------
        // calculate dataIndex
        switch traversal {
        case .normal:
            dataIndex = getDataIndex(row, col)

        case .padded:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getDataIndex(row - rowBounds.before,
                                         col - colBounds.before)
            }

        case .repeated:
            dataIndex = getRepeatedDataIndex(row, col)

        case .paddedRepeated:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getRepeatedDataIndex(row - rowBounds.before,
                                                 col - colBounds.before)
            }
        }
    }

    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        next.viewIndex += 1
        next.col += 1
        if next.col == colBounds.viewExtent {
            next.col = 0
            next.row += 1
        }
        next.computeDataIndex()
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
        next.computeDataIndex()
        return next
    }
}
