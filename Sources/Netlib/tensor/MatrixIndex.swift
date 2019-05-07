//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndex {
    // properties
    public let bounds: TensorBounds
    public var viewIndex: Int
    public var dataIndex: Int
    public var isPad: Bool

    public var row: Int
    public var col: Int
    public let isPadded: Bool
    public let isRepeated: Bool
    
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        bounds = view.createTensorBounds()
        isPadded = view.isPadded
        isRepeated = view.isRepeated
        row = position.r
        col = position.c
        isPad = false
        
        if isPadded {
            viewIndex = 0
            dataIndex = 0

        } else if isRepeated {
            viewIndex = 0
            dataIndex = 0

        } else {
            viewIndex = row * bounds[0].viewExtent + col * bounds[0].viewStride
            dataIndex = row * bounds[0].dataExtent + col * bounds[0].dataStride
        }
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
    public func advanced(by n: Int) -> MatrixIndex {
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
