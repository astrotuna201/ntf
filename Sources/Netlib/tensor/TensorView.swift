//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// TensorView protocol
/// A TensorView object is the primary interface for working with data in
/// the app and on various devices. Specialized shaped instances such as
/// Vector, Matrix, Volume, etc.. adopt this protocol. They will generically
/// be referred to as tensors after this.
///
/// Data can be safely accessed on the app thread and asynchronously on
/// device streams without the user needing be concerned with synchronization.
///
/// When a tensor is created, no memory is allocated until the first time
/// access is requested. The location of the access determines where the
/// buffer is created. No host shadow buffer is created. So temporary tensors
/// on local discrete devices or remote hosts can be freely created and
/// manipulated without any host resources being used, or data being transited
/// to the target device.
///
/// Data replication and synchronization are transparent to the user.
///
/// TensorViews are references to data and respect copy on write semantics,
/// locally and on device. Many operations can be performed with zero copy.
///
/// Data repeating (broadcasting) and padding are instrinsic features
///
public protocol TensorView: Logging, Equatable, DefaultInitializer {
    //--------------------------------------------------------------------------
    /// The type of scalar referenced by the view
    associatedtype Scalar: DefaultInitializer
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView

    //--------------------------------------------------------------------------
    // Properties that should be user readonly begin with _xyz, and accessor
    // functions with correct access are exposed as protocol extensions.
    // this gives full access to protocol default implementations.

    /// the shape of the actual underlying data. If `dataShape.extents` do not
    /// match `shape.extents` then the data is repeated (broadcast) across
    /// all dimensions during iteration and indexing.
    /// If `dataShape` is nil, then it equals `shape`
    var dataShape: DataShape { get }
    /// used internally when obtaining write access to manage
    /// multi-threaded writes without causing `tensorData` copy on write.
    var isShared: Bool { get }
    /// `true` if the view projects padded or repeated data
    var isVirtual: Bool { get }
    /// specifies an amount of padding before and after each dimension used
    /// only during indexing and iteration. It is not reflected in the `shape`
    /// of the view or part of subview creation. It is passed
    /// as a parameter to iterators. It is not inherited by subviews.
    var padding: [Padding] { get }
    /// the scalar value to be returned for indexes with padding regions
    var padValue: Scalar { get }
    /// the virtual shape of the view used for indexing
    /// if `shape` and `dataShape` are not equal, then `dataShape` is repeated
    var shape: DataShape { get }
    /// class reference to the underlying byte buffer
    var tensorData: TensorData { get set }
    /// the linear element offset where the view begins
    var viewDataOffset: Int { get set }

    //--------------------------------------------------------------------------
    /// create an empty view
    init()
    
    /// create a repeating view of other
    init(extents: [Int], repeating other: Self,
         padding: [Padding]?, padValue: Scalar?)
    
    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         padding: [Padding]?,
         padValue: Scalar?,
         tensorData: TensorData?,
         viewDataOffset: Int,
         isShared: Bool,
         scalars: [Scalar]?)
    
    /// determines if the view holds a unique reference to the underlying
    /// TensorData array
    mutating func isUniqueReference() -> Bool

    /// performs a dynamic rank reduction by removing extents of 1
    /// along the specified axes
    func squeezed(axes: [Int]?) -> NDTensor<Scalar>
}

//==============================================================================
/// IndexScalar
/// The data type used for tensors that contain tensor spatial index values
public typealias IndexScalar = Int32

public typealias ScalarConformance = DefaultInitializer

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    // public property accessors
    /// `true` if the scalars are contiguosly arranged in memory
    var isContiguous: Bool { return dataShape.isContiguous }
    /// is `true` if the last data access caused the view's underlying
    /// tensorData object to be copied.
    /// Used primarily for debugging and unit testing
    var lastAccessMutatedView: Bool { return tensorData.lastAccessMutatedView }
    /// the name of the view, which can optionally be set to aid in debugging
    var name: String { return tensorData.name }
    /// the number of dimensions in the view
    var rank: Int { return dataShape.rank }
    
    //--------------------------------------------------------------------------
    /// empty
    init() {
        self.init(shape: DataShape(),
                  dataShape: DataShape(),
                  name: nil,
                  padding: nil,
                  padValue: nil,
                  tensorData: TensorData(),
                  viewDataOffset: 0,
                  isShared: false,
                  scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// repeated view
    init(extents: [Int],
         repeating other: Self,
         padding: [Padding]? = nil,
         padValue: Scalar? = nil) {

        self.init(shape: DataShape(extents: extents),
                  dataShape: other.shape,
                  name: other.name,
                  padding: padding,
                  padValue: padValue,
                  tensorData: other.tensorData,
                  viewDataOffset: other.viewDataOffset,
                  isShared: other.isShared,
                  scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// initTensorData
    /// a helper to correctly initialize the tensorData object
    mutating func initTensorData(_ data: TensorData?,
                                 _ name: String?,
                                 _ scalars: [Scalar]?) {
        if let data = data {
            // this views existing data
            tensorData = data
        } else {
            // allocate backing tensorData
            assert(shape.isContiguous, "new views should have a dense shape")
            let tensorName = name ?? String(describing: Self.self)
            if let scalars = scalars {
                scalars.withUnsafeBytes {
                    tensorData = TensorData(buffer: $0, name: tensorName)
                }
            } else {
                tensorData = TensorData(type: Scalar.self,
                                        count: dataShape.elementSpanCount,
                                        name: tensorName)
            }
            assert(viewByteOffset + viewSpanByteCount <= tensorData.byteCount)
        }
    }
    
    // TODO: investigate need for this check
//    //--------------------------------------------------------------------------
//    /// shared memory
//    /// `true` if the underlying `tensorData` is being referenced by
//    /// `reference` views.
//    var isShared: Bool {
//        get { return _isShared }
//        set {
//            assert(!newValue || isShared || isUniqueReference(),
//                   "to set memory to shared it must already be shared or unique")
//            _isShared = newValue
//        }
//    }

    //--------------------------------------------------------------------------
    /// init<T>(shapedLike other:
    /// convenience initializer used by generics
    /// - Parameter other: the other object whose shape and logging to use
    init<T>(shapedLike other: T) where T: TensorView {
        self.init(shape: other.shape, dataShape: other.shape, name: nil,
                  padding: nil, padValue: nil,
                  tensorData: nil, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// init<S>(asScalar value:
    /// convenience initializer used by generics
    /// - Parameter value: the initial value to set
    init(_ value: Scalar) {
        // create scalar version of the shaped view type
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: nil,
                  padding: nil, padValue: nil,
                  tensorData: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
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
        let squeezedShape = shape.squeezed(axes: axes)
        return NDTensor<Scalar>(shape: squeezedShape,
                                dataShape: squeezedShape,
                                name: name,
                                padding: padding,
                                padValue: padValue,
                                tensorData: tensorData,
                                viewDataOffset: viewDataOffset,
                                isShared: isShared,
                                scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorData
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }
    
    //--------------------------------------------------------------------------
    /// viewByteOffset
    /// the byte offset into the `tensorData` buffer where this view begins
    var viewByteOffset: Int { return viewDataOffset * MemoryLayout<Scalar>.size}
    
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
        if lhs.tensorData === rhs.tensorData {
            // If they both reference the same tensorData then compare the views
            return
                lhs.viewDataOffset == rhs.viewDataOffset &&
                lhs.shape == rhs.shape
            
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
        return lhs.tensorData === rhs.tensorData && lhs == rhs
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorData if read-write access will cause mutation
    ///
    /// NOTE: this must be called from inside the accessQueue.sync block
    private mutating func copyIfMutates(using stream: DeviceStream?) throws {
        // for unit tests
        tensorData.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        diagnostic("\(mutationString) \(name)(\(tensorData.trackingId)) " +
            "elements[\(dataShape.elementCount)]",
            categories: [.dataCopy, .dataMutation])
        
        tensorData = try TensorData(withContentsOf: tensorData, using: stream)
        tensorData.lastAccessMutatedView = true
    }
    
    //--------------------------------------------------------------------------
    /// readOnly
    /// Returns a read only tensorData buffer pointer synced with the
    /// applicaiton thread. It's purpose is to be used by shaped subscript
    /// functions
    func readOnly() throws -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            tensorData.lastAccessMutatedView = false
            let buffer = try tensorData.readOnlyHostBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// readOnly(using stream:
    /// Returns a read only device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    func readOnly(using stream: DeviceStream) throws -> UnsafeRawPointer {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            tensorData.lastAccessMutatedView = false
            let buffer = try tensorData.readOnlyDevicePointer(using: stream)
            return buffer.advanced(by: viewDataOffset)
        }
    }
    
    /// readOnlyDeviceBuffer(using stream:
    /// Returns a read only device memory buffer synced with the specified
    /// stream. This version is used by accelerator APIs
    func readOnlyDeviceBuffer(using stream: DeviceStream) throws
        -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            tensorData.lastAccessMutatedView = false
            let buffer = try readOnly(using: stream)
                .bindMemory(to: Scalar.self,
                            capacity: dataShape.elementSpanCount)
            return UnsafeBufferPointer(start: buffer,
                                       count: dataShape.elementSpanCount)
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
        let queue = tensorData.accessQueue
        return try queue.sync {
            try copyIfMutates(using: nil)
            let buffer = try tensorData.readWriteHostBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// readWrite(using stream:
    /// Returns a read write device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    mutating func readWrite(using stream: DeviceStream) throws
        -> UnsafeMutableRawPointer {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            let buffer = try tensorData.readWriteDevicePointer(using: stream)
            return buffer.advanced(by: viewDataOffset)
        }
    }
    
    /// readWriteDeviceBuffer(using stream:
    /// Returns a read write device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    mutating func readWriteDeviceBuffer(using stream: DeviceStream) throws
        -> UnsafeMutableBufferPointer<Scalar> {
            let pointer = try readWrite(using: stream)
                .bindMemory(to: Scalar.self,
                            capacity: dataShape.elementSpanCount)
            return UnsafeMutableBufferPointer<Scalar>(
                start: pointer, count: dataShape.elementSpanCount)
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorData relative to this view
    private func createSubView(at offset: [Int], with extents: [Int],
                               padding: [Padding]?, padValue: Scalar?,
                               isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))

        // the subview offset is the current plus the offset of index
        let subViewOffset = viewDataOffset + shape.linearIndex(of: offset)
        let subViewShape = DataShape(extents: extents, strides: shape.strides)
        
        return Self.init(
            shape: subViewShape,
            dataShape: subViewShape,
            name: "\(name).subview",
            padding: padding,
            padValue: padValue,
            tensorData: tensorData,
            viewDataOffset: subViewOffset,
            isShared: isReference,
            scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorData relative to this view
    func view(at offset: [Int],
              extents: [Int],
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
        let flatShape = shape.flattened()
        return Self.init(shape: flatShape,
                         dataShape: flatShape,
                         name: name,
                         padding: padding,
                         padValue: padValue,
                         tensorData: tensorData,
                         viewDataOffset: viewDataOffset,
                         isShared: isShared,
                         scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// reference
    /// creation of a reference is for the purpose of reshaped write
    /// operations. Therefore the data will be copied before
    /// reference view creation if not uniquely held. References will not
    /// be checked on the resulting view when a write pointer is taken
    mutating func reference(using stream: DeviceStream? = nil) throws -> Self {
        // get the queue, if we reference it as a tensorData member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self.init(shape: shape,
                             dataShape: dataShape,
                             name: name,
                             padding: padding,
                             padValue: padValue,
                             tensorData: tensorData,
                             viewDataOffset: viewDataOffset,
                             isShared: true,
                             scalars: nil)
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
        let queue = tensorData.accessQueue
        
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
        let queue = tensorData.accessQueue
        
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
// map(to:
public extension Zip2Sequence {
    @inlinable func map<T>(to: inout T,
                           _ transform: ((Sequence1.Element, Sequence2.Element))
        throws -> T.Scalar) rethrows
        where T: TensorView {
        
        fatalError("not implemented yet")
    }
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

//==============================================================================
/// DefaultInitializer
public protocol DefaultInitializer {
    init()
}

extension Int8 : DefaultInitializer {}
extension UInt8 : DefaultInitializer {}
extension UInt16 : DefaultInitializer {}
extension Int16 : DefaultInitializer {}
extension UInt32 : DefaultInitializer {}
extension Int32 : DefaultInitializer {}
extension UInt : DefaultInitializer {}
extension Float : DefaultInitializer {}
extension Double : DefaultInitializer {}
extension Bool : DefaultInitializer {}
