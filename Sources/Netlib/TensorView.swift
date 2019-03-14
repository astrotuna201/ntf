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
    @noDerivative public let log: Log?
    @noDerivative public var logLevel = LogLevel.error
    @noDerivative public let nestingLevel = 0
    @noDerivative public var namePath: String = "TODO"

    // shape and shorthand accessors
    @noDerivative public let shape: TensorShape
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
    public init(shape: TensorShape,
                tensorData: TensorData<Scalar>? = nil,
                elementOffset: Int = 0,
                isShared: Bool = false,
                name: String? = nil,
                log: Log? = nil) {
        // assign
        let elementSize = MemoryLayout<Scalar>.size
        self.shape = shape
        self.isShared = isShared
        self.elementOffset = elementOffset
        self.viewByteOffset = elementOffset * elementSize
        self.viewByteCount = shape.elementSpanCount * elementSize
        self.log = log
        self.tensorData = tensorData ?? TensorData(
                log: log, elementCount: shape.elementCount, name: name)

        assert(viewByteOffset + viewByteCount <= self.tensorData.byteCount)
    }

    //------------------------------------
    // simple empty arrays
    public init() {
        log = nil
        shape = TensorShape()
        tensorData = TensorData()
        isShared = false
        elementOffset = 0
        viewByteCount = 0
        viewByteOffset = 0
    }

    public init(count: Int, log: Log? = nil) {
        self.init(shape: TensorShape(count), log: log)
    }

    public init(extents: [Int], log: Log? = nil) {
        self.init(shape: TensorShape(extents), log: log)
    }

    public init(_ extents: Int..., log: Log? = nil) {
        self.init(shape: TensorShape(extents), log: log)
    }

    //------------------------------------
    // from array
    public init(shape: TensorShape, scalars: [Scalar], log: Log? = nil) {
        let byteCount = scalars.count * MemoryLayout<Scalar>.size
        let dataPointer = scalars.withUnsafeBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { $0 }
        }
        self.init(shape: shape, start: dataPointer, count: byteCount, log: log)
    }

    public init(scalars: [Scalar], log: Log? = nil) {
        self.init(shape: TensorShape(scalars.count), scalars: scalars, log: log)
    }

    //------------------------------------
    // copy from pointer
    public init(shape: TensorShape, start: UnsafePointer<UInt8>, count: Int, log: Log? = nil) {
        let buffer = UnsafeBufferPointer(start: start, count: count)
        let tensorData = TensorData<Scalar>(log: log, buffer: buffer)
        self.init(shape: shape, tensorData: tensorData, log: log)
    }

    //--------------------------------------------------------------------------
    /// compose
    /// This function is used to join multiple views to create a higher rank
    /// shape. For example a set of 2D views can be joined to form a virtual
    /// 3D view that can be indexed by operators without copying any data.
    public func compose(with: TensorView...) -> TensorView {
        // TODO
        let shape = self
        return shape
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
