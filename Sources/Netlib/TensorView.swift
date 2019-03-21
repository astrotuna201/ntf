//******************************************************************************
//  Created by Edward Connell on 3/3/19.
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation
import TensorFlow

public struct TensorView<Scalar: TensorFlowScalar>: Differentiable, Logging {
    //--------------------------------------------------------------------------
    // properties
    @noDerivative private var tensorData: TensorData<Scalar>

    // logging
    @noDerivative public var logging: LogInfo?

    // shape and shorthand accessors
    @noDerivative public let shape: Shape
    @noDerivative public let elementOffset: Int
    @noDerivative public let viewByteOffset: Int
    @noDerivative public let viewByteCount: Int

    // convenience shorthand
    public var isContiguous: Bool { return shape.isContiguous }
    public var isEmpty: Bool { return shape.isEmpty }
    public var isScalar: Bool { return shape.isScalar }
    public var rank: Int { return shape.rank }

    public var items: Int { return shape.items }
    public var channels: Int { return shape.channels }
    public var depths: Int { return shape.depths }
    public var rows: Int { return shape.rows }
    public var cols: Int { return shape.cols }

    public var itemStride: Int { return shape.itemStride }
    public var channelStride: Int { return shape.channelStride }
    public var depthStride: Int { return shape.depthStride }
    public var rowStride: Int { return shape.rowStride }
    public var colStride: Int { return shape.colStride }

    //--------------------------------------------------------------------------
    // testing properties
    /// set any time the underlying TensorData is mutated
    @noDerivative public private(set) var lastAccessMutated = false
    /// determines is the view holds a unique reference to the underlying
    /// TensorData array
    public mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }

    //--------------------------------------------------------------------------
    // name
    @noDerivative public var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }

    //--------------------------------------------------------------------------
    // shared memory
    @noDerivative public var isShared: Bool {
        willSet {
            assert(!newValue || isShared || isUniqueReference(),
                   "to set memory to shared it must already be shared or unique")
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    // fully specified
    public init(shape: Shape,
                tensorData: TensorData<Scalar>? = nil,
                elementOffset: Int = 0,
                isShared: Bool = false,
                name: String? = nil,
                logging: LogInfo? = nil) {
        // assign
        let elementSize = MemoryLayout<Scalar>.size
        self.shape = shape
        self.isShared = isShared
        self.elementOffset = elementOffset
        self.logging = logging
        self.viewByteOffset = elementOffset * elementSize
        self.viewByteCount = shape.elementSpanCount * elementSize
        self.tensorData = tensorData ?? TensorData(
                logging: logging, elementCount: shape.elementCount, name: name)

        assert(viewByteOffset + viewByteCount <= self.tensorData.byteCount)
    }

    //--------------------------------------------------------------------------
    // create a sub view
    public func view(at offset: [Int], with extents: [Int]) -> TensorView {
        // the view created will have the same isShared state as the parent
        return createSubView(at: offset, with: extents, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    // view(item:
    public func view(item: Int) -> TensorView    {
        return viewItems(at: item, count: 1)
    }
    
    //--------------------------------------------------------------------------
    // viewItems
    // the view created will have the same isShared state as the parent
    public func viewItems(at offset: Int, count: Int) -> TensorView {
        var index: [Int]
        let viewExtents: [Int]
        if rank == 1 {
            index = [offset]
            viewExtents = [count]
        } else {
            index = [offset] + [Int](repeating: 0, count: rank - 1)
            viewExtents = [count] + shape.extents.suffix(from: 1)
        }
        
        return createSubView(at: index, with: viewExtents, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    // createSubView
    private func createSubView(at offset: [Int], with extents: [Int],
                               isReference: Bool) -> TensorView {
        // validate
        assert(extents[0] <= shape.items)
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset,
                              shape: Shape(extents: extents,
                                                 layout: shape.layout)))
        
        let eltOffset = elementOffset + shape.linearIndex(of: offset)
        let viewShape = Shape(extents: extents,
                                    layout: shape.layout,
                                    strides: shape.strides,
                                    colMajor: shape.isColMajor)
        
        return TensorView(shape: viewShape, tensorData: tensorData,
                          elementOffset: eltOffset, isShared: isReference)
    }

    //--------------------------------------------------------------------------
    // shuffle
    private static func shuffle(items count: Int, itemStride: Int) -> [Int] {
        var index = (0..<count).map { $0 * itemStride }
        var shuffledIndex = [Int]()
        while !index.isEmpty {
            let selected = Int.random(in: 0..<index.count)
            shuffledIndex.append(index[selected])
            index[selected] = index.last!
            index.removeLast()
        }
        return shuffledIndex
    }
}

//==============================================================================
// initializers
extension TensorView {
    //--------------------------------------------------------------------------
    // empty tensor
    public init() {
        elementOffset = 0
        isShared = false
        logging = nil
        shape = Shape()
        tensorData = TensorData()
        viewByteCount = 0
        viewByteOffset = 0
    }
    
    //--------------------------------------------------------------------------
    // implicitly zero is the default
    public init(count: Int, logging: LogInfo? = nil) {
        self.init(shape: Shape(count), logging: logging)
    }
    
    public init(extents: [Int], logging: LogInfo? = nil) {
        self.init(shape: Shape(extents), logging: logging)
    }
    
    public init(_ extents: Int..., logging: LogInfo? = nil) {
        self.init(shape: Shape(extents), logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // explicitly zero
    public init(zeros extents: [Int], logging: LogInfo? = nil) {
        self.init(shape: Shape(extents), logging: logging)
    }
    
    public init(zeros extents: Int..., logging: LogInfo? = nil) {
        self.init(shape: Shape(extents), logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // from Array
    public init(shape: Shape, scalars: [Scalar], logging: LogInfo? = nil) {
        let buffer = scalars.withUnsafeBufferPointer { $0 }
        let tensorData = TensorData<Scalar>(logging: logging, buffer: buffer)
        self.init(shape: shape, tensorData: tensorData, logging: logging)
    }
    
    public init(scalars: [Scalar], logging: LogInfo? = nil) {
        self.init(shape: Shape(scalars.count), scalars: scalars, logging: logging)
    }
    
    public init(extents: Int..., scalars: [Scalar], logging: LogInfo? = nil) {
        self.init(shape: Shape(extents: extents),
                  scalars: scalars, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // copy from host buffer pointer
    public init(shape: Shape,
                start: UnsafePointer<Scalar>, count: Int,
                logging: LogInfo? = nil) {
        let buffer = UnsafeBufferPointer(start: start, count: count)
        let tensorData = TensorData<Scalar>(logging: logging, buffer: buffer)
        self.init(shape: shape, tensorData: tensorData, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // type cast
    public init(_ other: TensorView<Bool>) {
        // TODO
        self = TensorView()
    }

    //--------------------------------------------------------------------------
    public func scalarized() throws -> Scalar {
        return try tensorData.roHostBuffer()[0]
    }
}
