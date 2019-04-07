//******************************************************************************
//  Created by Edward Connell on 3/3/16.
//  Copyright © 2016 Edward Connell. All rights reserved.
//
import Foundation
import TensorFlow

//==============================================================================
// DataShapeExtent
public struct DataShapeExtent: Equatable, Codable {
    /// void space before the shape
    let before: Int
    /// void space after the shape
    let after: Int
    /// the extent of the data
    let dataExtent: Int
    /// the stride of the data
    let datastride: Int
    /// the virtual extent of the data including before and after
    let virtualExtent: Int
    /// the virtual stride of the data including before and after
    let virtualStride: Int
}

//==============================================================================
// DataShape
public struct DataShape: Equatable, Codable {
    //--------------------------------------------------------------------------
    // properties
    /// The extent of the data associated with this shape in each dimension
    public let dataExtents: [Int]
    /// The extent of the indexable range in each dimension. Usually they will
    /// be equal to the `dataExtent`. Iterators and direct indexing will map
    /// memory indexes modulo the `dataExtent`. This is to support transparent
    /// broadcasting and to simplify operator implementation.
    public let extents: [Int]
    /// The distance to the next element for each dimension
    public let strides: [Int]
    /// The dense number of elements defined by the shape
    public let elementCount: Int
    /// The sparse number of elements spanned by the shape
    public let elementSpanCount: Int
    /// `true` if the underlying data is arranged in column major order
    public let isColMajor: Bool
    /// the index of the last dimension
    public let lastDimension: Int

    //--------------------------------------------------------------------------
    // computed properties

    /// `true` if the underlying data for the whole shape has a stride of 1.
    public var isContiguous: Bool { return elementCount == elementSpanCount }
    /// `true` if the shape has zero elements
    public var isEmpty: Bool { return elementCount == 0 }
    /// `true` if the shape is readonly because it is a virtual shape
    public var isReadOnly: Bool { return false } // TODO fix
    /// `true` if the shape has one element
    public var isScalar: Bool { return elementCount == 1 }
    /// the number of sahpe extents
    public var rank: Int { return dataExtents.count }
    /// the number of items in extent 0
    public var items: Int { return dataExtents[0] }

    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// - Parameter extents: extent of the shape in each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    /// - Parameter isColMajor: if `true`, the underlying `tensorData` is
    ///   interpreted as being layed out in column major order but will
    ///   present itself as if it is row major. This is to support importing
    ///   of row major data, such as matrices from Matlab or Octave.
    ///   It is assumed the last two extents are rows and columns.
    public init(extents: [Int],
                strides: [Int]? = nil,
                dataExtents: [Int]? = nil,
                padding: [Padding]? = nil,
                isColMajor: Bool = false) {
        // validate and assign
        assert(strides == nil || strides?.count == extents.count)
        let rank = extents.count
        self.lastDimension = rank - 1
        self.isColMajor = isColMajor

        // extents
        self.extents = extents
        self.dataExtents = dataExtents ?? extents
        self.elementCount = extents.count == 0 ? 0 : extents.reduce(1, *)

        // strides
        if let userStrides = strides {
            self.strides = userStrides
        } else if isColMajor {
            var cmExtent = extents
            cmExtent.swapAt(rank-1, rank-2)
            var cmStrides = DataShape.denseStrides(for: cmExtent)
            cmStrides.swapAt(rank-1, rank-2)
            self.strides = cmStrides
        } else {
            self.strides = DataShape.denseStrides(for: extents)
        }
        elementSpanCount = DataShape.spanCount(for: extents,
                                               with: self.strides)
    }

    //--------------------------------------------------------------------------
    /// Initialize with an array literal representing the shape extents.
    /// The rank of the tensor is the number of extents.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(arrayLiteral elements: Int...) {
        self.init(extents: elements)
    }

    //--------------------------------------------------------------------------
    /// Initialize with variadic elements representing the shape extents.
    /// The rank of the tensor is the number of elements.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(_ elements: Int...) {
        self.init(extents: elements)
    }

    //--------------------------------------------------------------------------
    /// Initialize with an array representing the shape extents.
    /// The rank of the tensor is the number of elements.
    /// - Parameter elements: The shape extents.
    @inlinable @inline(__always)
    public init(_ elements: [Int]) {
        self.init(extents: elements)
    }

    //--------------------------------------------------------------------------
    /// returns a dense version of self
    public var dense: DataShape {
        guard !isContiguous else { return self }
        return DataShape(extents: dataExtents, isColMajor: isColMajor)
    }
    
    //--------------------------------------------------------------------------
    // denseStrides
    private static func denseStrides(for extents: [Int]) -> [Int] {
        guard extents.count > 0 else { return [] }
        var strides = [Int](repeating: 1, count: extents.count)
        for index in (1..<extents.count).reversed() {
            strides[index-1] = extents[index] * strides[index]
        }
        return strides
    }

    //--------------------------------------------------------------------------
    /// makePositive(indices:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference indexes from the end of `extents`
    /// This ensures they are resolved to positive values.
    public func makePositive(indices: [Int]) -> [Int] {
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
        guard extents.count > 0 else { return 0 }
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
        default: result = zip(dataExtents, strides).reduce(0) { $0 + $1.0 * $1.1 }
        }
        assert(result <= elementSpanCount)
        return result
    }

    public func linearIndex(of index: Int...) -> Int {
        return linearIndex(of: index)
    }

    //--------------------------------------------------------------------------
    // contains
    // TODO maybe we don't need these
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
    /// squeezed(axes:
    /// performs a rank reduction by removing dimensions with an extent of 1
    /// - Parameter axes: the axes to squeeze. `nil` implies all axes.
    /// - Returns: the new data shape
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    public func squeezed(axes: [Int]? = nil) -> DataShape {
        assert(axes == nil || axes!.count <= rank)
        let axesSet = Set(makePositive(indices: axes ?? [Int](0..<rank)))
        var newExtents = [Int]()
        var newStrides = [Int]()
        
        for axis in 0..<rank
            where !(dataExtents[axis] == 1 && axesSet.contains(axis)) {
                
            newExtents.append(dataExtents[axis])
            newStrides.append(strides[axis])
        }
        
        return DataShape(extents: newExtents, strides: newStrides,
                         isColMajor: isColMajor)
    }

    //--------------------------------------------------------------------------
    /// transposed(with permutations:
    /// Returns a new data shape where the extents and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    /// - Precondition: Each value in `permutations` must be in the range
    ///   `-rank..<rank`
    public func transposed(with permutations: [Int]? = nil) -> DataShape {
        assert(rank > 1)
        assert(permutations == nil || permutations?.count == rank)
        var newExtents = [Int]()
        var newStrides = [Int]()

        // determine the new extents and strides
        if let perm = permutations {
            let mapping = makePositive(indices: perm)
            for index in 0..<rank {
                newExtents[index] = dataExtents[mapping[index]]
                newStrides[index] = strides[mapping[index]]
            }
        } else {
            // simple swap
            newExtents = dataExtents
            newStrides = strides
            newExtents.swapAt(rank-1, rank-2)
            newStrides.swapAt(rank-1, rank-2)
        }

        // return the new shape
        return DataShape(extents: newExtents, strides: newStrides,
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
        case 1: extent = [dataExtents[0], elementCount / dataExtents[0]]
        default:
            extent = [Int](dataExtents.prefix(upTo: axis)) +
                [dataExtents.suffix(from: axis).reduce(1, *)] +
                [Int](repeating: 1, count: rank - axis - 1)
        }
        return DataShape(extent)
    }
}

//==============================================================================
// pad
public struct Padding: Equatable, Codable {
    let before: Int
    let after: Int
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
        self = TensorFlow.TensorShape(shape.dataExtents.map { Int32($0) })
    }
}
