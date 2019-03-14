//******************************************************************************
//  Created by Edward Connell on 3/3/19.
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

public struct TensorView<Scalar>: Logging {
    //--------------------------------------------------------------------------
    // properties
    public var tensorData: TensorData<Scalar>

    // logging
    public let log: Log?
    public var logLevel = LogLevel.error
    public let nestingLevel = 0
    public var namePath: String = "TODO"

    // shape and shorthand accessors
    public let shape: TensorShape
    public let elementOffset: Int
    public let viewByteOffset: Int
    public let viewByteCount: Int

    //--------------------------------------------------------------------------
    // testing properties
    /// set any time the underlying TensorData is mutated
    public private(set) var lastAccessMutated = false
    /// determines is the view holds a unique reference to the underlying
    /// TensorData array
    public mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }

    //--------------------------------------------------------------------------
    // name
    public var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
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

    //--------------------------------------------------------------------------
    // empty view
    public init() {
        log = nil
        shape = TensorShape()
        tensorData = TensorData()
        isShared = false
        elementOffset = 0
        viewByteCount = 0
        viewByteOffset = 0
    }

    //--------------------------------------------------------------------------
    // shared memory
    public var isShared: Bool {
        willSet {
            assert(!newValue || isShared || isUniqueReference(),
                "to set memory to shared it must already be shared or unique")
        }
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
