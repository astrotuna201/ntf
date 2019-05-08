//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// VectorIndex
public struct VectorIndex: TensorIndex {
    // properties
    public var viewIndex: Int = 0
    public var dataIndex: Int = 0
    public var isPad: Bool = false
    
    // local properties
    public let traversal: TensorTraversal
    public let bounds: ExtentBounds
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: VectorPosition) where T: TensorView {
        bounds = view.createTensorBounds()[0]
        traversal = view.traversal
        viewIndex = position
        computeDataIndex()
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        bounds = view.createTensorBounds()[0]
        traversal = view.traversal
        viewIndex = view.shape.padded(with: view.padding).elementCount
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable @inline(__always)
    public mutating func computeDataIndex() {
        func getDataIndex(_ i: Int) -> Int {
            return i * bounds.dataStride
        }
        
        func getRepeatedDataIndex(_ i: Int) -> Int {
            return (i % bounds.dataExtent) * bounds.dataStride
        }
        
        func testIsPad() -> Bool {
            return viewIndex < bounds.before || viewIndex >= bounds.after
        }
        
        //----------------------------------
        // calculate dataIndex
        switch traversal {
        case .normal:
            dataIndex = getDataIndex(viewIndex)
            
        case .padded:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getDataIndex(viewIndex - bounds.before)
            }
            
        case .repeated:
            dataIndex = getRepeatedDataIndex(viewIndex)
            
        case .paddedRepeated:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getRepeatedDataIndex(viewIndex - bounds.before)
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> VectorIndex {
        var next = self
        next.viewIndex += 1
        next.computeDataIndex()
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> VectorIndex {
        guard n != 1 else { return increment() }
        var next = self
        next.viewIndex += n
        next.computeDataIndex()
        return next
    }
}

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

//==============================================================================
/// VolumeIndex
public struct VolumeIndex: TensorIndex {
    // properties
    public var viewIndex: Int = 0
    public var dataIndex: Int = 0
    public var isPad: Bool = false
    
    // local properties
    public let traversal: TensorTraversal
    public let depBounds: ExtentBounds
    public let rowBounds: ExtentBounds
    public let colBounds: ExtentBounds
    public var dep: Int
    public var row: Int
    public var col: Int
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: VolumePosition) where T: TensorView {
        let bounds = view.createTensorBounds()
        depBounds = bounds[0]
        rowBounds = bounds[1]
        colBounds = bounds[2]
        dep = position.d
        row = position.r
        col = position.c
        traversal = view.traversal
        viewIndex =
            dep * depBounds.viewStride +
            row * rowBounds.viewStride +
            col * colBounds.viewStride
        computeDataIndex()
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        let bounds = view.createTensorBounds()
        depBounds = bounds[0]
        rowBounds = bounds[1]
        colBounds = bounds[2]
        dep = 0
        row = 0
        col = 0
        traversal = view.traversal
        viewIndex = view.shape.padded(with: view.padding).elementCount
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable @inline(__always)
    public mutating func computeDataIndex() {
        func getDataIndex(_ d: Int, _ r: Int, _ c: Int) -> Int {
            return d * depBounds.dataStride +
                r * rowBounds.dataStride + c * colBounds.dataStride
        }
        
        func getRepeatedDataIndex(_ d: Int, _ r: Int, _ c: Int) -> Int {
            return (d % depBounds.dataExtent) * depBounds.dataStride +
                (r % rowBounds.dataExtent) * rowBounds.dataStride +
                (c % colBounds.dataExtent) * colBounds.dataStride
        }
        
        func testIsPad() -> Bool {
            return dep < depBounds.before || dep >= depBounds.after ||
                row < rowBounds.before || row >= rowBounds.after ||
                col < colBounds.before || col >= colBounds.after
        }
        
        //----------------------------------
        // calculate dataIndex
        switch traversal {
        case .normal:
            dataIndex = getDataIndex(dep, row, col)
            
        case .padded:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getDataIndex(dep - depBounds.before,
                                         row - rowBounds.before,
                                         col - colBounds.before)
            }
            
        case .repeated:
            dataIndex = getRepeatedDataIndex(dep, row, col)
            
        case .paddedRepeated:
            isPad = testIsPad()
            if !isPad {
                dataIndex = getRepeatedDataIndex(dep - depBounds.before,
                                                 row - rowBounds.before,
                                                 col - colBounds.before)
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> VolumeIndex {
        var next = self
        next.viewIndex += 1
        next.col += 1
        if next.col == colBounds.viewExtent {
            next.col = 0
            next.row += 1
            if next.row == rowBounds.viewExtent {
                next.row = 0
                next.dep += 1
            }
        }
        next.computeDataIndex()
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> VolumeIndex {
        guard n != 1 else { return increment() }
        var next = self

        // update the depth, row, and column positions
        var jump = n.quotientAndRemainder(dividingBy: depBounds.viewExtent)
        let quotient = jump.quotient
        next.dep += quotient
        
        jump = quotient.quotientAndRemainder(dividingBy: rowBounds.viewExtent)
        next.row += jump.quotient
        next.col += jump.remainder
        
        // now set the indexes
        next.computeDataIndex()
        return next
    }
}
