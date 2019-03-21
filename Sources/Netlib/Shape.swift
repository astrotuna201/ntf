//******************************************************************************
//  Created by Edward Connell on 3/3/16.
//  Copyright © 2016 Edward Connell. All rights reserved.
//
import Foundation

public struct Shape: Equatable {
    //--------------------------------------------------------------------------
    // properties
    /// The interpretation of each channel in the shape
    public let channelLayout: ChannelLayout
    /// The extent of the shape in each dimension
    public let extents: [Int]
    /// The dense number of elements defined by the shape
    public let elementCount: Int
    /// The sparse number of elements spanned by the shape
    public let elementSpanCount: Int
    /// True if rows and cols are arranged in column major order
    public let isColMajor: Bool
    /// The interpretation of each extent in the shape
    public let layout: TensorLayout
    /// The distance to the next element for each dimension
    public let strides: [Int]

    // convenience shorthand
    public var isContiguous: Bool { return elementCount == elementSpanCount }
    public var isEmpty: Bool { return elementCount == 0 }
    public var isScalar: Bool { return layout == .scalar }
    public var rank: Int { return extents.count }

    public var items: Int { return extents[layout.nAxis] }
    public var channels: Int { return extents[layout.cAxis] }
    public var depths: Int { return extents[layout.dAxis] }
    public var rows: Int { return extents[layout.hAxis] }
    public var cols: Int { return extents[layout.wAxis] }

    public var itemStride: Int { return strides[layout.nAxis] }
    public var channelStride: Int { return strides[layout.cAxis] }
    public var depthStride: Int { return strides[layout.dAxis] }
    public var rowStride: Int { return strides[layout.hAxis] }
    public var colStride: Int { return strides[layout.wAxis] }

    //--------------------------------------------------------------------------
    /// Initialize with all options
    /// - Parameter extents: extent of the shape in each dimension
    /// - Parameter layout: defines the interpretation of each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    /// - Parameter colMajor: if `true` it allows normal indexing of imported
    ///             row major data, such as matrices from Matlab or Octave
    public init(extents: [Int],
                layout: TensorLayout? = nil,
                channelLayout: ChannelLayout = .any,
                strides: [Int]? = nil,
                colMajor: Bool = false) {
        // validate and assign
        assert(strides == nil || strides?.count == extents.count)
        let rank = extents.count
        self.channelLayout = channelLayout
        self.extents = extents
        self.elementCount = extents.reduce(1, *)
        self.layout = layout ?? TensorLayout(defaultForRank: rank)
        self.isColMajor = colMajor

        // strides
        if let userStrides = strides {
            self.strides = userStrides
        } else if colMajor {
            var cmExtent = extents
            cmExtent.swapAt(self.layout.hAxis, self.layout.wAxis)
            var cmStrides = Shape.denseStrides(for: cmExtent)
            cmStrides.swapAt(self.layout.hAxis, self.layout.wAxis)
            self.strides = cmStrides
        } else {
            self.strides = Shape.denseStrides(for: extents)
        }
        elementSpanCount = Shape.spanCount(for: extents,
                                                 with: self.strides)
    }

    /// Initialize with an array literal representing the shape extents.
    /// The rank of the tensor is the number of extents.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(arrayLiteral elements: Int...) {
        self.init(extents: elements)
    }

    /// Initialize with variadic elements representing the shape extents.
    /// The rank of the tensor is the number of elements.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(_ elements: Int...) {
        self.init(extents: elements)
    }

    /// Initialize with an array representing the shape extents.
    /// The rank of the tensor is the number of elements.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(_ elements: [Int]) {
        self.init(extents: elements)
    }

    //--------------------------------------------------------------------------
    // denseStrides
    private static func denseStrides(for extents: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: extents.count)
        for index in (1..<extents.count).reversed() {
            strides[index-1] = extents[index] * strides[index]
        }
        return strides
    }

    //--------------------------------------------------------------------------
    // spanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of elements defined by the extent of this view
    // The span of the extent is the linear index of the last index + 1
    private static func spanCount(for extents: [Int],
                                  with strides: [Int]) -> Int {
        return zip(extents, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 } + 1
    }
    
    //--------------------------------------------------------------------------
    // linearIndex
    //    returns the linear element index
    public func linearIndex(of index: [Int]) -> Int {
        assert(rank > 0 && index.count == rank)
        var result: Int
        switch rank {
        case 1: result = index[0]
        case 2: result = index[0] * strides[0] + index[1] * strides[1]
        default: result = index[0] * strides[0] + index[1] * strides[1] +
            index[2] * strides[2] + index[3] * strides[3]
        }
        assert(result <= elementSpanCount)
        return result
    }

    //--------------------------------------------------------------------------
    // contains
    public func contains(index: [Int]) -> Bool {
        assert(index.count == rank, "rank mismatch")
        return linearIndex(of: index) <= elementSpanCount
    }
    
    public func contains(shape: Shape) -> Bool {
        assert(shape.rank == rank, "rank mismatch")
        return shape.elementSpanCount <= elementSpanCount
    }
    
    public func contains(offset: [Int], shape: Shape) -> Bool {
        assert(offset.count == rank && shape.rank == rank, "rank mismatch")
        return linearIndex(of: offset) + shape.elementSpanCount <= elementSpanCount
    }
}

//==============================================================================
// TensorLayout
public enum TensorLayout: Int {
    // warning: don't rearrange without also updating axis mapping below
    case scalar, vector, matrix, volume, nchw, nhwc, ncdhw, ndhwc

    // axis mapping                 s  ve m  vo nc nh nc nd
    public var nAxis: Int { return [0, 0, 0, 0, 0, 0, 0, 0][rawValue] }
    public var cAxis: Int { return [0, 0, 0, 0, 1, 3, 1, 4][rawValue] }
    public var dAxis: Int { return [0, 0, 0, 0, 0, 0, 2, 1][rawValue] }
    public var hAxis: Int { return [0, 0, 0, 1, 2, 1, 3, 2][rawValue] }
    public var wAxis: Int { return [0, 0, 1, 2, 3, 2, 4, 3][rawValue] }

    public init(defaultForRank rank: Int) {
        self = [.scalar, .vector, .matrix, .volume, .nchw, .ncdhw][rank]
    }
}

//==============================================================================
/// ChannelLayout
/// This is used to label channel to aid in automated format conversion
public enum ChannelLayout {
    // other
    case any
    // image
    case gray, grayAlpha, rgb, rgba
    // etc...
}

