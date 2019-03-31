//******************************************************************************
//  Created by Edward Connell on 3/3/16.
//  Copyright © 2016 Edward Connell. All rights reserved.
//
import Foundation
import TensorFlow

public struct TensorView<Scalar>: Logging, Equatable
where Scalar: AnyNumeric & TensorFlowScalar {
    //--------------------------------------------------------------------------
    // properties
    private var tensorData: TensorData<Scalar>

    // logging
    public var logging: LogInfo?

    // shape and shorthand accessors
    public let shape: DataShape
    public let viewOffset: Int

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
    // shared memory
    public var isShared: Bool {
        willSet {
            assert(!newValue || isShared || isUniqueReference(),
                   "to set memory to shared it must already be shared or unique")
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    // fully specified
    public init(shape: DataShape,
                tensorData: TensorData<Scalar>? = nil,
                viewOffset: Int = 0,
                isShared: Bool = false,
                name: String? = nil,
                logging: LogInfo? = nil) {
        // assign
        self.shape = shape
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.logging = logging
        self.tensorData = tensorData ?? TensorData(
                elementCount: shape.elementCount, logging: logging, name: name)

        assert(viewOffset + shape.elementCount <= self.tensorData.elementCount)
    }

    //--------------------------------------------------------------------------
    // helpers to obtain pointers to constants 0 and 1 to support Cuda calls
    private var _zero = Scalar(any: 0)
    public lazy var zeroPointer: UnsafeRawPointer = {
        return UnsafeRawPointer(withUnsafePointer(to: &_zero) { $0 })
    }()
    
    private var _one = Scalar(any: 1)
    public lazy var onePointer: UnsafeRawPointer = {
        return UnsafeRawPointer(withUnsafePointer(to: &_one) { $0 })
    }()

    //--------------------------------------------------------------------------
    // Equal values
    public static func == (lhs: TensorView<Scalar>,
                           rhs: TensorView<Scalar>) -> Bool {
        if lhs.tensorData === rhs.tensorData {
            // If they both reference the same tensorData then compare the views
            return lhs.viewOffset == rhs.viewOffset && lhs.shape == rhs.shape

        } else if lhs.shape.extents == rhs.shape.extents {
            // if the extents are equal then compare values
            // TODO use indexing
            fatalError("Not implemented")
        } else {
            return false
        }
    }
    
    //--------------------------------------------------------------------------
    // Equal references
    public static func === (lhs: TensorView<Scalar>,
                            rhs: TensorView<Scalar>) -> Bool {
        return lhs.tensorData === rhs.tensorData && lhs == rhs
    }
    
    //--------------------------------------------------------------------------
    /// isFinite
    /// This is primarily used for debugging
    ///
    /// Returns: true if all values are finite
    public func isFinite() throws -> Bool {
        guard !Scalar.isFiniteType else { return true }
        // TODO use DataShape indexing
        fatalError("Not implemented")
    }
    
    //--------------------------------------------------------------------------
    // copyIfMutates
    //  Note: this should be called from inside the accessQueue.sync block
    private mutating func copyIfMutates(using stream: DeviceStream? = nil) throws {
        // for unit tests
        lastAccessMutated = false
        guard !isShared && !isUniqueReference() else { return }
        
        lastAccessMutated = true
        if willLog(level: .diagnostic) == true {
            diagnostic("""
                \(mutationString) \(logging?.namePath ?? "")
                (\(tensorData.trackingId))  elements: \(tensorData.elementCount)
                """, categories: [.dataCopy, .dataMutation])
        }
        
        tensorData = try TensorData(withContentsOf: tensorData, using: stream)
    }
    
    //--------------------------------------------------------------------------
    // Read only buffer access
    public func ro() throws -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            return try tensorData.roHostBuffer()
        }
    }
    
    // this version is for accelerator APIs
    public func ro(using stream: DeviceStream) throws -> UnsafeRawPointer {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            let buffer = try tensorData.roDevicePointer(using: stream)
            return buffer.advanced(by: viewOffset)
        }
    }

    //--------------------------------------------------------------------------
    // Read Write buffer access
    public mutating func rw() throws -> UnsafeMutableBufferPointer<Scalar> {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            try copyIfMutates()
            return try tensorData.rwHostBuffer()
        }
    }
    
    // this version is for accelerator APIs
    public mutating func rw(using stream: DeviceStream) throws
        -> UnsafeMutableRawPointer {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            let buffer = try tensorData.rwDevicePointer(using: stream)
            return buffer.advanced(by: viewOffset)
        }
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
                              shape: DataShape(extents: extents,
                                                 layout: shape.layout)))
        
        let eltOffset = viewOffset + shape.linearIndex(of: offset)
        let viewShape = DataShape(extents: extents,
                                  layout: shape.layout,
                                  strides: shape.strides,
                                  isColMajor: shape.isColMajor)
        
        return TensorView(shape: viewShape,
                          tensorData: tensorData,
                          viewOffset: eltOffset,
                          isShared: isReference,
                          logging: logging)
    }

    //--------------------------------------------------------------------------
    // flattened
    public func flattened(axis: Int = 0) -> TensorView {
        return createFlattened(axis: axis, isShared: isShared)
    }
    
    //--------------------------------------------------------------------------
    // createFlattened
    private func createFlattened(axis: Int, isShared: Bool) -> TensorView {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }

        return TensorView(shape: shape.flattened(),
                          tensorData: tensorData,
                          viewOffset: viewOffset,
                          isShared: isShared,
                          logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // reference
    // creation of a reference is for the purpose of reshaped write
    // operations. Therefore the data will be copied before
    // reference view creation if not uniquely held. References will not
    // be checked on the resulting view when a write pointer is taken
    public mutating func reference(using stream: DeviceStream?) throws -> TensorView {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return TensorView(shape: shape,
                              tensorData: tensorData,
                              viewOffset: viewOffset,
                              isShared: true,
                              logging: logging)
        }
    }
    
    //--------------------------------------------------------------------------
    // referenceView
    public mutating func referenceView(offset: [Int], extents: [Int],
                                       using stream: DeviceStream?) throws -> TensorView {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createSubView(at: offset, with: extents, isReference: true)
        }
    }
    
    //--------------------------------------------------------------------------
    // referenceFlattened
    public mutating func referenceFlattened(axis: Int = 0,
                                            using stream: DeviceStream?) throws -> TensorView {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createFlattened(axis: axis, isShared: true)
        }
    }
    
    //--------------------------------------------------------------------------
    // shuffle
    // TODO
    private static func shuffle(items count: Int, itemStride: Int) -> [Int] {
        var index = (0..<count).map { $0 * itemStride }
        var shuffledIndex = [Int]()
        while !index.isEmpty {
            let selected = Int.random(in: 0..<index.count)
            shuffledIndex.append(index[selected])
            index[selected] = index.last!
            index.removeLast()
        }
        fatalError("Not implemented")
    }
}

//==============================================================================
// initializers
extension TensorView {
    //--------------------------------------------------------------------------
    // empty tensor
    public init() {
        isShared = false
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
        viewOffset = 0
    }
    
    //--------------------------------------------------------------------------
    // This will perform a Scalar type cast and return a dense view
    /// Perform element-wise type conversion
    @inlinable @inline(__always)
    init<OtherScalar>(_ other: TensorView<OtherScalar>,
                      using deviceStream: DeviceStream? = nil) throws {
        //    self = Raw.cast(other)
        // omit strides argument to create dense shape
        let dense = DataShape(extents: other.shape.extents,
                              layout: other.shape.layout,
                              channelLayout: other.shape.channelLayout,
                              isColMajor: other.shape.isColMajor)
        
        // create output view and bundle parameters
        let stream = deviceStream ?? _ThreadLocal.value.defaultStream
        self = TensorView(shape: dense)
        try stream.cast(from: other, to: &self)
    }
    
    //--------------------------------------------------------------------------
    // implicitly zero is the default
    public init(count: Int, logging: LogInfo? = nil) {
        self.init(shape: DataShape(count), logging: logging)
    }
    
    public init(extents: [Int], logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents), logging: logging)
    }
    
    public init(extents: Int..., logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents), logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // explicitly zero
    //    By default arrays are cleared to 0
    public init(zeros extents: [Int], logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents), logging: logging)
    }
    
    public init(zeros extents: Int..., logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents), logging: logging)
    }
    
    public init(zeros shape: DataShape, logging: LogInfo? = nil) {
        self.init(shape: shape, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // from Array
    public init(shape: DataShape, scalars: [Scalar], logging: LogInfo? = nil) {
        let buffer = scalars.withUnsafeBufferPointer { $0 }
        let tensorData = TensorData<Scalar>(logging: logging, buffer: buffer)
        self.init(shape: shape, tensorData: tensorData, logging: logging)
    }
    
    public init(scalars: [Scalar], logging: LogInfo? = nil) {
        self.init(shape: DataShape(scalars.count), scalars: scalars, logging: logging)
    }
    
    public init(extents: Int..., scalars: [Scalar], logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents: extents),
                  scalars: scalars, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // copy from host buffer pointer
    public init<T>(_ value: T, logging: LogInfo? = nil) where T: AnyNumeric {
        self.init(extents: 1, logging: logging)
        try! rw()[0] = Scalar(any: value)
    }
    
    //--------------------------------------------------------------------------
    // copy from host buffer pointer
    public init(shape: DataShape,
                start: UnsafePointer<Scalar>, count: Int,
                logging: LogInfo? = nil) {
        let buffer = UnsafeBufferPointer(start: start, count: count)
        let tensorData = TensorData<Scalar>(logging: logging, buffer: buffer)
        self.init(shape: shape, tensorData: tensorData, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    // fill
    public init(repeating repeatedValue: Scalar, shape: DataShape) {
        fatalError("Not implemented")
    }

    //--------------------------------------------------------------------------
    // legacy
    public init(legacy tensor: Tensor<Scalar>) {
        let shape = DataShape(legacy: tensor.shape)
        self = tensor.array.withUnsafeBufferPointer {
            return TensorView(shape: shape, start: $0.baseAddress!,
                              count: $0.count)
        }
    }

    //--------------------------------------------------------------------------
    public func scalar() throws -> Scalar {
        return try tensorData.roHostBuffer()[0]
    }
}

//==============================================================================
// initializers
public extension TensorFlow.Tensor where Scalar: AnyTensorFlowScalar {
    /// create a dense copy of other and perform type conversion
    init<T: AnyTensorFlowScalar>(_ view: Netlib.TensorView<T>,
                                 using stream: DeviceStream? = nil) throws {

        let denseView = try TensorView<Scalar>(view, using: stream)
        self = Tensor<Scalar>(shape: TensorFlow.TensorShape(denseView.shape),
                              scalars: try denseView.ro())
    }
}
