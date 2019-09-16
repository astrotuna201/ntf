//******************************************************************************
//  Created by Edward Connell on 5/4/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation


//==============================================================================
/// ScalarIndex
public struct ScalarIndex: TensorIndexing {
    // properties
    public var bufferIndex: Int = 0
    
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
public struct VectorIndex: TensorIndexing {
    // properties
    public var bufferIndex: Int
    public let stride: Int
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: VectorPosition) where T: TensorView {
        stride = view.shape.strides[0]
        bufferIndex = position
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        stride = view.shape.strides[0]
        bufferIndex = view.shape.elementCount
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> VectorIndex {
        var next = self
        next.bufferIndex += stride
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> VectorIndex {
        guard n != 1 else { return increment() }
        var next = self
        next.bufferIndex += n * stride
        return next
    }
}

//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndexing {
    // properties
    public var bufferIndex: Int = 0
    public let rowExtent: Int
    public let rowStride: Int
    public let colExtent: Int
    public let colStride: Int
    public var row: Int
    public var col: Int
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        rowExtent = view.shape.extents[0]
        rowStride = view.shape.strides[0]
        row = position.r
        colExtent = view.shape.extents[1]
        colStride = view.shape.strides[1]
        col = position.c
        bufferIndex = computeBufferIndex()
    }

    public init<T>(endOf view: T) where T: TensorView {
        rowExtent = view.shape.extents[0]
        rowStride = view.shape.strides[0]
        row = rowExtent
        colExtent = view.shape.extents[1]
        colStride = view.shape.strides[1]
        col = 0
        bufferIndex = view.shape.elementCount
    }

    //--------------------------------------------------------------------------
    /// computeBufferIndex
    @inlinable @inline(__always)
    public func computeBufferIndex() -> Int {
        return row * rowStride + col * colStride
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> MatrixIndex {
        var next = self
        next.col += 1
        if next.col == colExtent {
            next.col = 0
            next.row += 1
            next.bufferIndex = next.computeBufferIndex()
        } else {
            next.bufferIndex += colStride
        }
        return next
    }

    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable @inline(__always)
    public func advanced(by n: Int) -> MatrixIndex {
        guard n != 1 else { return increment() }
        let jump = n.quotientAndRemainder(dividingBy: rowExtent)
        var next = self
        next.row += jump.quotient
        next.col += jump.remainder
        next.bufferIndex = next.computeBufferIndex()
        return next
    }
}

//==============================================================================
/// VolumeIndex
public struct VolumeIndex: TensorIndexing {
    // properties
    public var bufferIndex: Int = 0
    public let depExtent: Int
    public let depStride: Int
    public let rowExtent: Int
    public let rowStride: Int
    public let colExtent: Int
    public let colStride: Int
    public var dep: Int
    public var row: Int
    public var col: Int

    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: VolumePosition) where T: TensorView {
        depExtent = view.shape.extents[0]
        depStride = view.shape.strides[0]
        dep = position.d
        rowExtent = view.shape.extents[1]
        rowStride = view.shape.strides[1]
        row = position.r
        colExtent = view.shape.extents[2]
        colStride = view.shape.strides[2]
        col = position.c
        computeBufferIndex()
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        depExtent = view.shape.extents[0]
        depStride = view.shape.strides[0]
        dep = view.shape.extents[0]
        rowExtent = view.shape.extents[1]
        rowStride = view.shape.strides[1]
        row = 0
        colExtent = view.shape.extents[2]
        colStride = view.shape.strides[2]
        col = 0
        computeBufferIndex()
    }
    
    //--------------------------------------------------------------------------
    /// computeBufferIndex
    @inlinable @inline(__always)
    public mutating func computeBufferIndex() {
        bufferIndex = dep * depStride + row * rowStride + col * colStride
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> VolumeIndex {
        var next = self
        next.col += 1
        if next.col == colExtent {
            next.col = 0
            next.row += 1
            if next.row == rowExtent {
                next.row = 0
                next.dep += 1
            }
            next.computeBufferIndex()
        } else {
            next.bufferIndex += colStride
        }
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
        var jump = n.quotientAndRemainder(dividingBy: depExtent)
        let quotient = jump.quotient
        next.dep += quotient
        
        jump = quotient.quotientAndRemainder(dividingBy: rowExtent)
        next.row += jump.quotient
        next.col += jump.remainder
        
        // now set the index
        next.computeBufferIndex()
        return next
    }
}

//==============================================================================
/// NDIndex
public struct NDIndex: TensorIndexing {
    // properties
    public var bufferIndex: Int = 0
    public var position: NDPosition
    public let extents: [Int]
    public let strides: [Int]
    
    //--------------------------------------------------------------------------
    // initializers
    public init<T>(view: T, at position: NDPosition) where T: TensorView {
        extents = view.shape.extents
        strides = view.shape.strides
        self.position = position
        computeBufferIndex()
    }
    
    public init<T>(endOf view: T) where T: TensorView {
        position = [Int](repeating: 0, count: view.rank)
        position[0] = view.extents[0]
        bufferIndex = view.shape.elementCount
        extents = view.shape.extents
        strides = view.shape.strides
    }
    
    //--------------------------------------------------------------------------
    /// computeBufferIndex
    @inlinable @inline(__always)
    public mutating func computeBufferIndex() {
        bufferIndex = zip(position, strides).reduce(0) {
            $0 + $1.0 * $1.1
        }
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable @inline(__always)
    public func increment() -> NDIndex {
        var next = self

        // increment the last dimension
        func nextPosition(for dim: Int) {
            next.position[dim] += 1
            if next.position[dim] == extents[dim] && dim > 0 {
                next.position[dim] = 0
                nextPosition(for: dim - 1)
            }
        }
        nextPosition(for: extents.count - 1)
        
        // this should be rethought for the simple incremental case instead
        // of full recompute
        next.computeBufferIndex()
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
        for dim in (1..<extents.count).reversed() {
            jump = distance.quotientAndRemainder(dividingBy: extents[dim])
            next.position[dim] += jump.quotient
            distance = jump.remainder
            if dim == 1 {
                next.position[0] += jump.remainder
            }
        }
        next.computeBufferIndex()
        return next
    }
}
