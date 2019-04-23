//******************************************************************************
//  Created by Edward Connell on 3/3/16.
//  Copyright © 2016 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorShape
public struct DataShape: Equatable, Codable {
    //--------------------------------------------------------------------------
    // properties
    /// The dense number of elements defined by the shape
    public let elementCount: Int
    /// The sparse number of elements spanned by the shape
    public let elementSpanCount: Int
    /// The extent of the shape in each dimension
    public let extents: [Int]
    /// the index of the last dimension
    public let lastDimension: Int
    /// The distance to the next element for each dimension
    public let strides: [Int]

    //--------------------------------------------------------------------------
    // computed properties
    /// `true` if the underlying data for the whole shape has a stride of 1.
    public var isContiguous: Bool { return elementCount == elementSpanCount }
    /// `true` if the shape has zero elements
    public var isEmpty: Bool { return elementCount == 0 }
    /// `true` if the shape has one element
    public var isScalar: Bool { return elementCount == 1 }
    /// the number of sahpe extents
    public var rank: Int { return extents.count }
    /// the number of items in extent 0
    public var items: Int { return extents[0] }

    //--------------------------------------------------------------------------
    // empty shape
    public init() {
        extents = []
        strides = []
        lastDimension = 0
        elementCount = 0
        elementSpanCount = 0
    }

    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// - Parameter extents: extent of the shape in each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    public init(extents: [Int], strides: [Int]? = nil) {
        // validate
        assert(extents.count > 0, "use init() to create empty shape")
        assert(strides == nil || strides?.count == extents.count,
               "the stride count must equal the extent count")
        // init properties
        self.extents = extents
        self.lastDimension = extents.count - 1
        self.elementCount = extents.count == 0 ? 0 : extents.reduce(1, *)
        self.strides = strides ?? DataShape.denseStrides(for: extents)
        elementSpanCount = DataShape.spanCount(for: extents, with: self.strides)
    }

    //--------------------------------------------------------------------------
    /// returns a dense version of self
    public var dense: DataShape {
        guard !isContiguous else { return self }
        return DataShape(extents: extents)
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
        default: result = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
        }
        assert(result <= elementSpanCount)
        return result
    }

    public func linearIndex(of index: Int...) -> Int {
        return linearIndex(of: index)
    }

    //--------------------------------------------------------------------------
    /// contains
    /// used primarily for asserts
    public func contains(offset: [Int]) -> Bool {
        assert(offset.count == rank, "rank mismatch")
        return linearIndex(of: offset) <= elementSpanCount
    }
    
    public func contains(shape: DataShape) -> Bool {
        assert(shape.rank == rank, "rank mismatch")
        return shape.elementSpanCount <= elementSpanCount
    }
    
    public func contains(offset: [Int], extents: [Int]) -> Bool {
        assert(offset.count == rank && extents.count == rank, "rank mismatch")
        let span = linearIndex(of: offset) +
            DataShape.spanCount(for: extents, with: strides)
        return span <= elementSpanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
    public func columnMajor() -> DataShape {
        // compute column major strides for the last 2 dimensions
        var cmExtent = extents
        cmExtent.swapAt(rank-1, rank-2)
        var cmStrides = DataShape.denseStrides(for: cmExtent)
        cmStrides.swapAt(rank-1, rank-2)
        return DataShape(extents: extents, strides: cmStrides)
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
            where !(extents[axis] == 1 && axesSet.contains(axis)) {
                
            newExtents.append(extents[axis])
            newStrides.append(strides[axis])
        }
        
        return DataShape(extents: newExtents, strides: newStrides)
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
        return DataShape(extents: newExtents, strides: newStrides)
    }

    //--------------------------------------------------------------------------
    // flattened
    public func flattened(axis: Int = 0) -> DataShape {
        assert(isContiguous, "Cannot flatten strided data")
        assert(axis < rank)

        // create a new flat view
        var flatExtents: [Int]
        switch axis {
        case 0: flatExtents = [elementCount]
        case 1: flatExtents = [extents[0], elementCount / extents[0]]
        default:
            flatExtents = [Int](extents.prefix(upTo: axis)) +
                [extents.suffix(from: axis).reduce(1, *)] +
                [Int](repeating: 1, count: rank - axis - 1)
        }
        return DataShape(extents: flatExtents)
    }
    
    //--------------------------------------------------------------------------
    // padded
    public func padded(with padding: [Padding]) -> DataShape {
        // get the padding and set an increment if there is more than one
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0
        var paddedExtents = [Int]()
        for i in 0..<rank {
            paddedExtents.append(
                extents[i] + padding[padIndex].before + padding[padIndex].after)
            padIndex += padIncrement
        }
        return DataShape(extents: paddedExtents)
    }
}

//==============================================================================
// pad
public struct Padding: Equatable, Codable {
    let before: Int
    let after: Int

    public init(before: Int, after: Int) {
        self.before = before
        self.after = after
    }
    
    public init(_ both: Int) {
        self.before = both
        self.after = both
    }
}

////==============================================================================
//// Legacy TensorFlow.TensorShape
//public extension DataShape {
//    init(legacy shape: TensorFlow.TensorShape) {
//        self.init(extents: shape.dimensions.map { Int($0) })
//    }
//}
//
//public extension TensorFlow.TensorShape {
//    init(_ shape: Netlib.DataShape) {
//        self = TensorFlow.TensorShape(shape.extents)
//    }
//}
