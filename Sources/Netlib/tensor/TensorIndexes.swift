//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// ScalarIndex
public struct ScalarIndex: TensorIndex {
    // properties
    public var viewIndex: Int = 0
    public var dataIndex: Int = 0
    public var isPad: Bool = false
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: ScalarPosition) where T: TensorView {}
    public init<T>(endOf view: T) where T: TensorView { }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> ScalarIndex { return self }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> ScalarIndex { return self }
}


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

//==============================================================================
/// NDIndex
public struct NDIndex: TensorIndex {
    // properties
    public var viewIndex: Int = 0
    public var dataIndex: Int = 0
    public var isPad: Bool = false
    
    // local properties
    public let traversal: TensorTraversal
    public let bounds: TensorBounds
    public var position: NDPosition
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: NDPosition) where T: TensorView {
        traversal = view.traversal
        // get the bounds and exapand postion if needed
        let bounds = view.createTensorBounds()
        if position.count == 1 {
            self.position = [Int](repeating: position[0], count: bounds.count)
        } else {
            self.position = position
        }
        self.bounds = bounds
        
        // compute the initial view index
        viewIndex = zip(self.position, self.bounds).reduce(0) {
            $0 + $1.0 * $1.1.viewStride
        }
        computeDataIndex()
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        bounds = view.createTensorBounds()
        position = [Int](repeating: 0, count: bounds.count)
        traversal = view.traversal
        viewIndex = view.shape.padded(with: view.padding).elementCount
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable @inline(__always)
    public mutating func computeDataIndex() {
        func getDataIndex(_ p: [Int]) -> Int {
            return zip(p, bounds).reduce(0) {
                $0 + $1.0 * $1.1.viewStride
            }
        }
        
        func getRepeatedDataIndex(_ p: [Int]) -> Int {
            return zip(p, bounds).reduce(0) {
                $0 + ($1.0 % $1.1.dataExtent) * $1.1.viewStride
            }
        }
        
        func testIsPad() -> Bool {
            for dim in 0..<bounds.count {
                if position[dim] < bounds[dim].before ||
                    position[dim] >= bounds[dim].after {
                    return true
                }
            }
            return false
        }
        
        //----------------------------------
        // calculate dataIndex
        switch traversal {
        case .normal:
            dataIndex = getDataIndex(position)
            
        case .padded:
            isPad = testIsPad()
            if !isPad {
                let padPos = zip(position, bounds).map { $0.0 - $0.1.before }
                dataIndex = getDataIndex(padPos)
            }
            
        case .repeated:
            dataIndex = getRepeatedDataIndex(position)
            
        case .paddedRepeated:
            isPad = testIsPad()
            if !isPad {
                let padPos = zip(position, bounds).map { $0.0 - $0.1.before }
                dataIndex = getRepeatedDataIndex(padPos)
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> NDIndex {
        var next = self
        next.viewIndex += 1

        // increment the last dimension
        func nextPosition(for dim: Int) {
            next.position[dim] += 1
            if next.position[dim] == bounds[dim].viewExtent && dim > 0 {
                next.position[dim] = 0
                nextPosition(for: dim - 1)
            }
        }
        nextPosition(for: bounds.count - 1)
        
        next.computeDataIndex()
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> NDIndex {
        guard n != 1 else { return increment() }
        var next = self
        var distance = n
        
        var jump: (quotient: Int, remainder: Int)
        for dim in (1..<bounds.count).reversed() {
            jump = distance
                .quotientAndRemainder(dividingBy: bounds[dim].viewExtent)
            next.position[dim] += jump.quotient
            distance = jump.remainder
            if dim == 1 {
                next.position[0] += jump.remainder
            }
        }
        
        // now set the indexes
        next.computeDataIndex()
        return next
    }
}