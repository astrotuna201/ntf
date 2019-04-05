//******************************************************************************
//  Created by Edward Connell on 3/3/16.
//  Copyright © 2016 Edward Connell. All rights reserved.
//
import Foundation
import TensorFlow

public struct DataShape: Equatable, Codable {
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
    public var items: Int { return extents[0] }
    
    /// returns a dense version of self
    public var dense: DataShape {
        guard !isContiguous else { return self }
        return DataShape(extents: extents, layout: layout,
                         channelLayout: channelLayout, isColMajor: isColMajor)
    }

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
                isColMajor: Bool = false) {
        // validate and assign
        assert(strides == nil || strides?.count == extents.count)
        let rank = extents.count
        self.channelLayout = channelLayout
        self.extents = extents
        self.elementCount = extents.reduce(1, *)
        self.layout = layout ?? TensorLayout(defaultForRank: rank)
        self.isColMajor = isColMajor

        // strides
        if let userStrides = strides {
            self.strides = userStrides
        } else if isColMajor {
            var cmExtent = extents
            cmExtent.swapAt(self.layout.hAxis, self.layout.wAxis)
            var cmStrides = DataShape.denseStrides(for: cmExtent)
            cmStrides.swapAt(self.layout.hAxis, self.layout.wAxis)
            self.strides = cmStrides
        } else {
            self.strides = DataShape.denseStrides(for: extents)
        }
        elementSpanCount = DataShape.spanCount(for: extents,
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
    /// ensurePositive(indices:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference indexes from the end of `extents`
    /// This ensures they are resolved to positive values.
    public func ensurePositive(indices: [Int]) -> [Int] {
        return indices.map {
            assert(-rank..<rank ~= $0)
            return $0 < 0 ? $0 + rank : $0
        }
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
    /// linearIndex
    ///    returns the linear element index
    public func linearIndex(of index: [Int]) -> Int {
        assert(rank > 0 && index.count == rank)
        var result: Int
        switch rank {
        case 0: result = 0
        case 1: result = index[0]
        default: result = zip(extents, strides).reduce(0) { $0 + $1.0 * $1.1 }
        }
        assert(result <= elementSpanCount)
        return result
    }

    public func linearIndex(of index: Int...) -> Int {
        return linearIndex(of: index)
    }

    //--------------------------------------------------------------------------
    // contains
    public func contains(index: [Int]) -> Bool {
        assert(index.count == rank, "rank mismatch")
        return linearIndex(of: index) <= elementSpanCount
    }
    
    public func contains(shape: DataShape) -> Bool {
        assert(shape.rank == rank, "rank mismatch")
        return shape.elementSpanCount <= elementSpanCount
    }
    
    public func contains(offset: [Int], shape: DataShape) -> Bool {
        assert(offset.count == rank && shape.rank == rank, "rank mismatch")
        return linearIndex(of: offset) + shape.elementSpanCount <= elementSpanCount
    }

    //--------------------------------------------------------------------------
    // squeezed
    public func squeezed(axes: Set<Int>? = nil) -> DataShape {
        assert(axes == nil || axes!.count <= rank)
        var newExtents = [Int]()
        var newStrides = [Int]()
        let axes = axes ?? Set<Int>(0..<rank)
        for axis in 0..<rank where !(extents[axis] == 1 && axes.contains(axis)){
            newExtents.append(extents[axis])
            newStrides.append(strides[axis])
        }
        return DataShape(extents: newExtents, layout: layout,
                         channelLayout: channelLayout, strides: newStrides,
                         isColMajor: isColMajor)
    }

    //--------------------------------------------------------------------------
    /// transposed(with permutations:
    /// Returns a new data shape where the extents and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    public func transposed(with permutations: [Int]? = nil) -> DataShape {
        assert(rank > 1)
        assert(permutations == nil || permutations?.count == rank)
        var newExtents = [Int]()
        var newStrides = [Int]()

        // determine the new extents and strides
        if let perm = permutations {
            let mapping = ensurePositive(indices: perm)
            for index in 0..<rank {
                newExtents[index] = extents[mapping[index]]
                newStrides[index] = strides[mapping[index]]
            }
        } else {
            // simple swap
            newExtents = extents
            newStrides = strides
            newExtents.swapAt(rank-1, rank-2)
            newStrides.swapAt(rank-1, rank-2)
        }

        // return the new shape
        return DataShape(extents: newExtents, layout: .any,
                         channelLayout: channelLayout, strides: newStrides,
                         isColMajor: isColMajor)
    }

    //--------------------------------------------------------------------------
    // flattened
    public func flattened(axis: Int = 0) -> DataShape {
        assert(isContiguous, "Cannot reshape strided data")
        assert(axis < rank)

        // create a new flat view
        var extent: [Int]
        switch axis {
        case 0: extent = [elementCount]
        case 1: extent = [extents[0], elementCount / extents[0]]
        default:
            extent = [Int](extents.prefix(upTo: axis)) +
                [extents.suffix(from: axis).reduce(1, *)] +
                [Int](repeating: 1, count: rank - axis - 1)
        }
        return DataShape(extent)
    }
}

//==============================================================================
// Legacy TensorFlow.TensorShape
public extension DataShape {
    init(legacy shape: TensorFlow.TensorShape) {
        self.init(extents: shape.dimensions.map { Int($0) })
    }
}

public extension TensorFlow.TensorShape {
    init(_ shape: Netlib.DataShape) {
        self = TensorFlow.TensorShape(shape.extents.map { Int32($0) })
    }
}

//==============================================================================
// TensorLayout
public enum TensorLayout: Int, Codable {
    // warning: don't rearrange without also updating axis mapping below
    case any, scalar, vector, matrix, volume, nchw, nhwc, ncdhw, ndhwc

    // TODO: probably replace this scheme with specialized indexed types
    // axis mapping                 a  s  ve m  vo nc nh nc nd
    public var nAxis: Int { return [0, 0, 0, 0, 0, 0, 0, 0, 0][rawValue] }
    public var cAxis: Int { return [0, 0, 0, 0, 0, 1, 3, 1, 4][rawValue] }
    public var dAxis: Int { return [0, 0, 0, 0, 0, 0, 0, 2, 1][rawValue] }
    public var hAxis: Int { return [0, 0, 0, 0, 1, 2, 1, 3, 2][rawValue] }
    public var wAxis: Int { return [0, 0, 0, 1, 2, 3, 2, 4, 3][rawValue] }

    public init(defaultForRank rank: Int) {
        let defaults: [TensorLayout] =
            [.scalar, .vector, .matrix, .volume, .nchw, .ncdhw]
        self = rank >= defaults.count ? .any : defaults[rank]
    }
}

//==============================================================================
/// ChannelLayout
/// This is used to label channel to aid in automated format conversion
public enum ChannelLayout: Int, Codable {
    // other
    case any
    // image
    case gray, grayAlpha, rgb, rgba
    // etc...
}
