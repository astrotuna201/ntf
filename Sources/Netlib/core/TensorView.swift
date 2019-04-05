//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorView protocol
public protocol TensorView: AnyScalar, Logging, Equatable {
    //--------------------------------------------------------------------------
    /// The type of scalar referenced by the view
    associatedtype Scalar: AnyScalar
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView

    //--------------------------------------------------------------------------
    // Properties that should be user readonly begin with _xyz, and accessor
    // functions with correct access are exposed as protocol extensions.
    // this gives full access to protocol default implementations.
    
    /// during write access. Primarily to support multi-threaded writes
    var _isShared: Bool { get set }
    /// lastAccessMutated is `true` if the last data access caused the view
    /// to mutate, which causes the underlying tensorData object to be copied
    /// It's primary use is in debugging and unit testing
    var _lastAccessMutated: Bool { get set }
    /// the name of the view, which can optionally be set to aid in debugging
    var _name: String? { get set }
    /// the shape of the view
    var _shape: DataShape { get set }
    /// class reference to the underlying byte buffer
    var _tensorData: TensorData { get set }
    /// the linear element offset where the view begins
    var _viewOffset: Int { get set }
    /// the logging information for the view
    var logging: LogInfo? { get set }

    //--------------------------------------------------------------------------
    // initializers
    
    /// Fully specified initializer
    /// creates a new concrete view instance. This is required to enable
    /// extension methods to create typed return values
    init(shape: DataShape,
         tensorData: TensorData?,
         viewOffset: Int,
         isShared: Bool,
         name: String?,
         logging: LogInfo?)
    
    /// convenience initializer used to create result views in op functions
    init<T>(shapedLike other: T) where T: TensorView

    /// convenience initializer used to create type compatible tensors from
    /// from a scalar used in generic op functions.
    init(asScalar value: Scalar)
    
    //--------------------------------------------------------------------------
    // functions
    
    /// determines if the view holds a unique reference to the underlying
    /// TensorData array
    mutating func isUniqueReference() -> Bool

    /// performs a dynamic rank reduction by removing extents of 1
    /// along the specified axes
    func squeezed(axes: [Int]?) -> NDTensor<Scalar>
}

//------------------------------------------------------------------------------
/// The type used for indexing
public typealias TensorIndex = Int32

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    // public property accessors

    /// `true` if the scalars are densely packed in memory
    var isContiguous: Bool { return shape.isContiguous }
    /// `true` if the view contains zero elements
    var isEmpty: Bool { return shape.isEmpty }
    /// lastAccessMutated is `true` if the last data access caused the view
    /// to mutate, which causes the underlying tensorData object to be copied
    /// It's primary use is in debugging and unit testing
    var lastAccessMutated: Bool { return _lastAccessMutated }
    /// the number of dimensions in the view
    var rank: Int { return _shape.rank }
    /// the shape of the view
    var shape: DataShape { return _shape }
    
    //--------------------------------------------------------------------------
    /// shared memory
    /// `true` if the underlying `tensorData` is being referenced by
    /// `reference` views. 
    var isShared: Bool {
        get { return _isShared }
        set {
            assert(!newValue || isShared || isUniqueReference(),
                   "to set memory to shared it must already be shared or unique")
            _isShared = newValue
        }
    }

    //--------------------------------------------------------------------------
    /// name
    /// an optional view name used for logging
    var name: String {
        get { return _name ?? String(describing: self) }
        set {
            _name = newValue
            if !_tensorData.hasName { _tensorData.name = newValue }
        }
    }

    //--------------------------------------------------------------------------
    /// creates an empty view
    init() {
        self.init(shape: DataShape(), tensorData: TensorData(),
                  viewOffset: 0, isShared: false, name: nil, logging: nil)
    }

    //--------------------------------------------------------------------------
    /// init<T>(shapedLike other:
    /// convenience initializer used by generics
    /// - Parameter other: the other object whose shape and logging to use
    init<T>(shapedLike other: T) where T: TensorView {
        self.init(shape: other.shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: nil,
                  logging: other.logging)
    }
    
    //--------------------------------------------------------------------------
    /// init<S>(asScalar value:
    /// convenience initializer used by generics
    /// - Parameter value: the initial value to set
    init(asScalar value: Scalar) {
        // create scalar version of the shaped view type
        self.init(shape: DataShape(extents: [1]),
                  tensorData: nil, viewOffset: 0,
                  isShared: false, name: nil, logging: nil)
        // set the value
        try! readWrite()[0] = value
    }

    //--------------------------------------------------------------------------
    /// scalarValue
    /// - Returns: the single value in the tensor as a scalar
    func scalarValue() throws -> Scalar {
        assert(shape.elementCount == 1)
        return try readOnly()[0]
    }

    //--------------------------------------------------------------------------
    /// squeezed(axes:
    /// performs a rank reduction by removing dimensions with an extent of 1
    /// - Parameter axes: the axes to squeeze. `nil` implies all axes.
    /// - Returns: the new data shape
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    func squeezed(axes: [Int]? = nil) -> NDTensor<Scalar> {
        return NDTensor<Scalar>(
            shape: shape.squeezed(axes: axes),
            tensorData: _tensorData, viewOffset: _viewOffset,
            isShared: isShared, name: name, logging: logging)
    }

    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorData
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&_tensorData)
    }
    
    //--------------------------------------------------------------------------
    /// viewByteOffset
    /// the byte offset into the `tensorData` buffer where this view begins
    var viewByteOffset: Int { return _viewOffset * MemoryLayout<Scalar>.size }
    
    //--------------------------------------------------------------------------
    /// viewSpanByteCount
    /// the number of bytes in the `tensorData` spanned by this view
    var viewSpanByteCount: Int {
        return shape.elementSpanCount * MemoryLayout<Scalar>.size
    }

    //--------------------------------------------------------------------------
    /// isFinite
    /// `true` if all elements are finite values. Primarily used for debugging
    func isFinite() throws -> Bool {
        var isfiniteValue = true
        func check<T: AnyNumeric>(_ type: T.Type) throws {
            try readOnly().withMemoryRebound(to: AnyNumeric.self) {
                $0.forEach {
                    if !$0.isFiniteValue {
                        isfiniteValue = false
                    }
                }
            }
        }
        return isfiniteValue
    }
    
    //--------------------------------------------------------------------------
    /// Equal values
    /// performs an element wise value comparison
    static func == (lhs: Self, rhs: Self) -> Bool {
        if lhs._tensorData === rhs._tensorData {
            // If they both reference the same tensorData then compare the views
            return lhs._viewOffset == rhs._viewOffset && lhs.shape == rhs.shape
            
        } else if lhs.shape.extents == rhs.shape.extents {
            // if the extents are equal then compare values
            // TODO use Ops
            fatalError("Not implemented")
        } else {
            return false
        }
    }
    
    //--------------------------------------------------------------------------
    /// Equal references
    /// `true` if the views reference the same elements
    static func === (lhs: Self, rhs: Self) -> Bool {
        return lhs._tensorData === rhs._tensorData && lhs == rhs
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorData if read-write access will cause mutation
    ///
    /// NOTE: this must be called from inside the accessQueue.sync block
    private mutating func copyIfMutates(using stream: DeviceStream? = nil) throws {
        // for unit tests
        _lastAccessMutated = false
        guard !isShared && !isUniqueReference() else { return }
        
        _lastAccessMutated = true
        if willLog(level: .diagnostic) == true {
            diagnostic("""
                \(mutationString) \(logging?.namePath ?? "")
                (\(_tensorData.trackingId))  elements: \(shape.elementCount)
                """, categories: [.dataCopy, .dataMutation])
        }
        
        _tensorData = try TensorData(withContentsOf: _tensorData, using: stream)
    }
    
    //--------------------------------------------------------------------------
    /// readOnly
    /// Returns a read only tensorData buffer pointer synced with the
    /// applicaiton thread. It's purpose is to be used by shaped subscript
    /// functions
    func readOnly() throws -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        return try queue.sync {
            let buffer = try _tensorData.roHostRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// readOnly(using stream:
    /// Returns a read only device memory pointer synced with the specified
    /// stream. This version is by accelerator APIs
    func readOnly(using stream: DeviceStream) throws -> UnsafeRawPointer {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        return try queue.sync {
            let buffer = try _tensorData.roDevicePointer(using: stream)
            return buffer.advanced(by: _viewOffset)
        }
    }
    
    //--------------------------------------------------------------------------
    /// readWrite
    /// Returns a read write tensorData buffer pointer synced with the
    /// applicaiton thread. It's purpose is to be used by shaped subscript
    /// functions
    mutating func readWrite() throws -> UnsafeMutableBufferPointer<Scalar> {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        return try queue.sync {
            try copyIfMutates()
            let buffer = try _tensorData.rwHostMutableRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// readWrite(using stream:
    /// Returns a read write device memory pointer synced with the specified
    /// stream. This version is by accelerator APIs
    mutating func readWrite(using stream: DeviceStream) throws
        -> UnsafeMutableRawPointer {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            let buffer = try _tensorData.rwDevicePointer(using: stream)
            return buffer.advanced(by: _viewOffset)
        }
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorData relative to this view
    private func createSubView(at offset: [Int], with extents: [Int],
                               isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(extents[0] <= shape.extents[0])
        assert(shape.contains(offset: offset,
                              shape: DataShape(extents: extents,
                                               layout: shape.layout)))
        // find subview relative offset and shape
        let elementOffset = _viewOffset + shape.linearIndex(of: offset)
        let subViewShape = DataShape(extents: extents,
                                     layout: shape.layout,
                                     channelLayout: shape.channelLayout,
                                     strides: shape.strides,
                                     isColMajor: shape.isColMajor)
        
        return Self.init(shape: subViewShape,
                         tensorData: _tensorData,
                         viewOffset: elementOffset,
                         isShared: isReference,
                         name: "\(name).subview",
                         logging: logging)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorData relative to this view
    func view(at offset: [Int], with extents: [Int]) -> Self {
        // the view created will have the same isShared state as the parent
        return createSubView(at: offset, with: extents, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// viewItems
    /// Returns a view along extent[0] spanning all the other extents. It is
    /// used to simplify accessing a set of training samples.
    ///
    /// The view created will have the same isShared state as the parent
    func viewItems(at offset: Int, count: Int) -> Self {
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
    /// view(item:
    func view(item: Int) -> Self {
        return viewItems(at: item, count: 1)
    }
    
    //--------------------------------------------------------------------------
    /// flattened
    /// Returns a view reduced in rank depending on the axis selected
    func flattened(axis: Int = 0) -> Self {
        return createFlattened(axis: axis, isShared: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// createFlattened
    /// helper
    private func createFlattened(axis: Int, isShared: Bool) -> Self {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }
        
        // create flattened view
        return Self.init(shape: shape.flattened(),
                         tensorData: _tensorData, viewOffset: _viewOffset,
                         isShared: isShared, name: name, logging: logging)
    }
    
    //--------------------------------------------------------------------------
    /// reference
    /// creation of a reference is for the purpose of reshaped write
    /// operations. Therefore the data will be copied before
    /// reference view creation if not uniquely held. References will not
    /// be checked on the resulting view when a write pointer is taken
    mutating func reference(using stream: DeviceStream?) throws -> Self {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self.init(shape: shape,
                             tensorData: _tensorData, viewOffset: _viewOffset,
                             isShared: true, name: name, logging: logging)
        }
    }
    
    //--------------------------------------------------------------------------
    /// referenceView
    /// Creates a reference view relative to this view. Write operations will
    /// not cause mutation of tensorData. It's purpose is to support
    /// multi-threaded write operations
    mutating func referenceView(offset: [Int], extents: [Int],
                                using stream: DeviceStream?) throws -> Self {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createSubView(at: offset, with: extents, isReference: true)
        }
    }
    
    //--------------------------------------------------------------------------
    /// referenceFlattened
    /// Creates a flattened reference view relative to this view.
    /// Write operations will not cause mutation of tensorData.
    /// It's purpose is to support multi-threaded write operations
    mutating func referenceFlattened(axis: Int = 0,
                                     using stream: DeviceStream?) throws -> Self {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createFlattened(axis: axis, isShared: true)
        }
    }
}
