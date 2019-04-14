//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorView protocol
public protocol TensorView: Logging, Equatable {
    //--------------------------------------------------------------------------
    /// The type of scalar referenced by the view
    associatedtype Scalar
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView

    //--------------------------------------------------------------------------
    // Properties that should be user readonly begin with _xyz, and accessor
    // functions with correct access are exposed as protocol extensions.
    // this gives full access to protocol default implementations.

    // remove these
    /// specifies an amount of padding before and after each dimension used
    /// only during indexing and iteration. It is not reflected in the `shape`
    /// of the view or part of subview creation. It is passed
    /// as a parameter to iterators. It is not inherited by subviews.
    var padding: [Padding]? { get set }
    /// the scalar value to be returned for indexes with padding regions
    var padValue: Scalar? { get set }

    /// the shape of the actual underlying data. If `dataShape.extents` do not
    /// match `shape.extents` then the data is repeated (broadcast) across
    /// all dimensions during iteration and indexing.
    /// If `_dataShape` is nil, then it equals `shape`
    var _dataShape: DataShape? { get set }
    /// `true` if the shape is readonly because it is a virtual shape or if
    /// it references a read only memory buffer
    var _isReadOnly: Bool { get set }
    /// used internally when obtaining write access to manage
    /// multi-threaded writes without causing `tensorData` copy on write.
    var _isShared: Bool { get set }
    /// the name of the view, which can optionally be set to aid in debugging
    var _name: String? { get set }
    /// the virtual shape of the view used for indexing
    /// if `shape` and `dataShape` are not equal, then `dataShape` is repeated
    var _shape: DataShape { get set }
    /// class reference to the underlying byte buffer
    var _tensorData: TensorData { get set }
    /// the linear element offset where the view begins
    var _viewOffset: Int { get set }
    /// the logging information for the view
    var logging: LogInfo? { get set }

    //--------------------------------------------------------------------------
    // functions
    init()
    
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

    /// the shape of the actual underlying data. If `dataShape.extents` do not
    /// match `shape.extents` then the data is repeated (broadcast) across
    /// all dimensions during iteration and indexing.
    var dataShape: DataShape { return _dataShape ?? shape }
    /// `true` if the scalars are contiguosly arranged in memory
    var isContiguous: Bool { return dataShape.isContiguous }
    /// `true` if the view contains zero elements
    var isEmpty: Bool { return dataShape.isEmpty }
    /// `true` if the shape is readonly because it is a virtual shape or if
    /// it references a read only memory buffer
    var isReadOnly: Bool { return _isReadOnly }
    /// is `true` if the last data access caused the view's underlying
    /// tensorData object to be copied.
    /// Used primarily for debugging and unit testing
    var lastAccessMutatedView: Bool
    { return _tensorData.lastAccessMutatedView }
    /// the number of dimensions in the view
    var rank: Int { return dataShape.rank }
    /// the shape of the view
    var shape: DataShape { return _shape }
    /// the linear element offset where the view begins
    var viewOffset: Int { return _viewOffset }
    
    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// creates a new concrete view instance. This is required to enable
    /// extension methods to create typed return values
    init(shape: DataShape,
         dataShape: DataShape? = nil,
         tensorData: TensorData? = nil,
         viewOffset: Int = 0,
         padding: [Padding]? = nil,
         padValue: Scalar? = nil,
         isShared: Bool = false,
         name: String? = nil,
         logging: LogInfo? = nil)
    {
        self.init()
        _dataShape = dataShape
        _isShared = isShared
        _name = name
        _shape = shape
        _viewOffset = viewOffset
        self.padding = padding
        self.padValue = padValue
        self.logging = logging
        _tensorData = initTensorData(tensorData, shape, dataShape)
    }

    //--------------------------------------------------------------------------
    /// repeated view
    init(extents: [Int], repeating other: Self,
         padding: [Padding]? = nil, padValue: Scalar? = nil) {

        self.init(shape: DataShape(extents: extents),
                  dataShape: other.shape,
                  tensorData: other._tensorData,
                  viewOffset: other._viewOffset,
                  padding: padding,
                  padValue: padValue,
                  isShared: other._isShared,
                  name: other.name,
                  logging: other.logging)
    }

    //--------------------------------------------------------------------------
    /// initTensorData
    /// a helper to correctly initialize the tensorData object
    func initTensorData(_ param: TensorData?, _ shape: DataShape,
                        _ dataShape: DataShape?) -> TensorData {
        if let tensorData = param {
            // this views existing data
            return tensorData
        } else {
            // allocate backing tensorData
            assert(shape.isContiguous, "new views should have a dense shape")
            assert(dataShape == nil, "new views shouldn't specify a data shape")
            let tensorData = TensorData(type: Scalar.self,
                                        count: self.dataShape.elementSpanCount,
                                        logging: logging, name: name)
            
            assert(viewByteOffset + viewSpanByteCount <= tensorData.byteCount)
            return tensorData
        }
    }

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
        get { return _name ?? _tensorData.name }
        set {
            _name = newValue
            if !_tensorData.hasName { _tensorData.name = newValue }
        }
    }

    //--------------------------------------------------------------------------
    /// init<T>(shapedLike other:
    /// convenience initializer used by generics
    /// - Parameter other: the other object whose shape and logging to use
    init<T>(shapedLike other: T) where T: TensorView {
        self.init(shape: other.shape, logging: other.logging)
    }
    
    //--------------------------------------------------------------------------
    /// init<S>(asScalar value:
    /// convenience initializer used by generics
    /// - Parameter value: the initial value to set
    init(_ value: Scalar) {
        // create scalar version of the shaped view type
        self.init(shape: DataShape(extents: [1]))
        // set the value
        try! readWrite()[0] = value
    }

    //--------------------------------------------------------------------------
    /// - Returns: the padded extents for the view used for iteration
    func getPaddedExtents() -> [Int] {
        guard let padding = self.padding else { return shape.extents }
        let padIncrement = padding.count > 1 ? 1 : 0
        var padIndex = 0
        var padExtents = [Int]()
        
        for dim in 0..<rank {
            let span = padding[padIndex].before +
                shape.extents[dim] +
                padding[padIndex].after
            
            padExtents.append(span)
            padIndex += padIncrement
        }
        return padExtents
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
        return dataShape.elementSpanCount * MemoryLayout<Scalar>.size
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
        _tensorData.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        if willLog(level: .diagnostic) == true {
            diagnostic("""
                \(mutationString) \(logging?.namePath ?? "")
                (\(_tensorData.trackingId))  elements: \(dataShape.elementCount)
                """, categories: [.dataCopy, .dataMutation])
        }
        
        _tensorData = try TensorData(withContentsOf: _tensorData, using: stream)
        _tensorData.lastAccessMutatedView = true
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
            _tensorData.lastAccessMutatedView = false
            let buffer = try _tensorData.readOnlyHostBuffer()
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
            _tensorData.lastAccessMutatedView = false
            let buffer = try _tensorData.readOnlyDevicePointer(using: stream)
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
            let buffer = try _tensorData.readWriteHostBuffer()
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
            let buffer = try _tensorData.readWriteDevicePointer(using: stream)
            return buffer.advanced(by: _viewOffset)
        }
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorData relative to this view
    private func createSubView(at offset: [Int], with extents: [Int],
                               padding: [Padding]?, padValue: Scalar?,
                               isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(extents[0] <= shape.extents[0])
        assert(shape.contains(offset: offset,
                              shape: DataShape(extents: extents)))
        // find subview relative offset and shape
        let dataOffset = zip(offset, dataShape.extents).map { $0 % $1 }
        let elementOffset = _viewOffset + shape.linearIndex(of: dataOffset)
        let subViewShape = DataShape(extents: extents, strides: shape.strides)
        
        return Self.init(shape: subViewShape,
                         dataShape: dataShape,
                         tensorData: _tensorData,
                         viewOffset: elementOffset,
                         padding: padding,
                         padValue: padValue,
                         isShared: isReference,
                         name: "\(name).subview",
                         logging: logging)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorData relative to this view
    func view(at offset: [Int],
              with extents: [Int],
              padding: [Padding]? = nil,
              padValue: Scalar? = nil) -> Self {
        // the view created will have the same isShared state as the parent
        return createSubView(at: offset, with: extents,
                             padding: padding, padValue: padValue,
                             isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// viewItems
    /// Returns a view along the first dimension spanning all the others.
    /// It is used to simplify accessing a set of training samples.
    /// The view created will have the same isShared state as the parent
    func viewItems(at offset: Int,
                   count: Int,
                   padding: [Padding]? = nil,
                   padValue: Scalar? = nil) -> Self {

        let index, viewExtents: [Int]
        if rank == 1 {
            index = [offset]
            viewExtents = [count]
        } else {
            index = [offset] + [Int](repeating: 0, count: rank - 1)
            viewExtents = [count] + shape.extents.suffix(from: 1)
        }
        
        return createSubView(at: index, with: viewExtents,
                             padding: padding, padValue: padValue,
                             isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// view(item:
    func view(item: Int,
              padding: [Padding]? = nil,
              padValue: Scalar? = nil) -> Self {
        return viewItems(at: item, count: 1,
                         padding: padding, padValue: padValue)
    }
    
    //--------------------------------------------------------------------------
    /// flattened
    /// Returns a view reduced in rank depending on the axis selected
    func flattened(axis: Int = 0,
                   padding: [Padding]? = nil,
                   padValue: Scalar? = nil) -> Self {
        
        return createFlattened(axis: axis, isShared: isShared,
                               padding: padding, padValue: padValue)
    }
    
    //--------------------------------------------------------------------------
    /// createFlattened
    /// helper
    private func createFlattened(axis: Int,
                                 isShared: Bool,
                                 padding: [Padding]?,
                                 padValue: Scalar?) -> Self {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }
        
        // create flattened view
        return Self.init(shape: shape.flattened(),
                         dataShape: _dataShape,
                         tensorData: _tensorData,
                         viewOffset: _viewOffset,
                         padding: padding,
                         padValue: padValue,
                         isShared: isShared,
                         name: name,
                         logging: logging)
    }
    
    //--------------------------------------------------------------------------
    /// reference
    /// creation of a reference is for the purpose of reshaped write
    /// operations. Therefore the data will be copied before
    /// reference view creation if not uniquely held. References will not
    /// be checked on the resulting view when a write pointer is taken
    mutating func reference(using stream: DeviceStream? = nil) throws -> Self {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self.init(shape: shape, dataShape: _dataShape,
                             tensorData: _tensorData, viewOffset: _viewOffset,
                             padding: padding, padValue: padValue,
                             isShared: true,
                             name: name, logging: logging)
        }
    }
    
    //--------------------------------------------------------------------------
    /// referenceView
    /// Creates a reference view relative to this view. Write operations will
    /// not cause mutation of tensorData. It's purpose is to support
    /// multi-threaded write operations
    
    // TODO: maybe remove this if a subview view can correctly be taken
    // from a `reference` view
    mutating func referenceView(
        offset: [Int], extents: [Int],
        padding: [Padding]? = nil, padValue: Scalar? = nil,
        using stream: DeviceStream? = nil) throws -> Self {
        
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createSubView(at: offset, with: extents,
                                 padding: padding, padValue: padValue,
                                 isReference: true)
        }
    }
    
    //--------------------------------------------------------------------------
    /// referenceFlattened
    /// Creates a flattened reference view relative to this view.
    /// Write operations will not cause mutation of tensorData.
    /// It's purpose is to support multi-threaded write operations

    // TODO: maybe remove this if a subview view can correctly be taken
    // from a `reference` view
    mutating func referenceFlattened(
        axis: Int = 0,
        padding: [Padding]? = nil, padValue: Scalar? = nil,
        using stream: DeviceStream? = nil) throws -> Self {
        
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = _tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createFlattened(axis: axis, isShared: true,
                                   padding: padding, padValue: padValue)
        }
    }
}

public extension TensorView where Scalar: FloatingPoint {
    //--------------------------------------------------------------------------
    /// isFinite
    /// `true` if all elements are finite values. Primarily used for debugging
    func isFinite() throws -> Bool {
        for value in try readOnly() {
            if !value.isFinite {
                return false
            }
        }
        return true
    }
}

//==============================================================================
// PaddedView protocol
public protocol PaddedView: TensorView {
    /// specifies an amount of padding before and after each dimension used
    /// only during indexing and iteration. It is not reflected in the `shape`
    /// of the view or part of subview creation. It is passed
    /// as a parameter to iterators. It is not inherited by subviews.
    var padding: [Padding] { get set }
    /// the scalar value to be returned for indexes with padding regions
    var padValue: Scalar { get set }
}

//==============================================================================
// QuantizedView protocol
/// scalars are transformed from Scalar -> ViewedScalar during iteration
public protocol QuantizedView: TensorView {
    /// the scalar type stored by the view
    associatedtype Scalar
    /// the scalar type presented by the view
    associatedtype ViewedScalar
    
    /// the bias to apply during conversion
    var bias: ViewedScalar { get set }
    /// the scale to apply during conversion
    var scale: ViewedScalar { get set }
}

