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
public protocol TensorView: Logging, DefaultInitializer {
    //--------------------------------------------------------------------------
    /// The type of scalar referenced by the view
    associatedtype Scalar: DefaultInitializer
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView
    /// A tensor shape specific indexer used to calculate a data buffer
    /// index based on a view's spatial position
    associatedtype ViewIndex: TensorIndex
    
    //--------------------------------------------------------------------------
    // Properties that should be user readonly begin with _xyz, and accessor
    // functions with correct access are exposed as protocol extensions.
    // this gives full access to protocol default implementations.

    /// the shape of the actual underlying data. If `dataShape.extents` do not
    /// match `shape.extents` then the data is repeated (broadcast) across
    /// all dimensions during iteration and indexing.
    /// If `dataShape` is nil, then it equals `shape`
    var dataShape: DataShape { get }
    /// returns an index one past the end of the tensor used for collections
    var endIndex: ViewIndex { get }
    /// used internally when obtaining write access to manage
    /// multi-threaded writes without causing `tensorArray` copy on write.
    var isShared: Bool { get }
    /// specifies an amount of padding before and after each dimension used
    /// only during indexing and iteration. It is not reflected in the `shape`
    /// of the view or part of subview creation. It is passed
    /// as a parameter to iterators. It is not inherited by subviews.
    var padding: [Padding]? { get }
    /// the scalar value to be returned for indexes with padding regions
    var padValue: Scalar { get }
    /// the virtual shape of the view used for indexing
    /// if `shape` and `dataShape` are not equal, then `dataShape` is repeated
    var shape: DataShape { get }
    /// returns the first tensor index used for collections
    var startIndex: ViewIndex { get }
    /// class reference to the underlying byte buffer
    var tensorArray: TensorArray { get set }
    /// the indexing traversal procedure to use
    var traversal: TensorTraversal { get }
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
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         isShared: Bool,
         scalars: [Scalar]?)
}

//==============================================================================
/// IndexScalar
/// The data type used for tensors that contain tensor spatial index values
public typealias IndexScalar = Int32

//==============================================================================
/// traversal
public enum TensorTraversal: Int32 {
    case normal, padded, repeated, paddedRepeated
}

public func initTraversal(_ padding: [Padding]?, _ isRepeated: Bool)
    -> TensorTraversal
{
    if padding != nil {
        return isRepeated ? .paddedRepeated : .padded
    } else if isRepeated {
        return .repeated
    } else {
        return .normal
    }
}

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    // public property accessors
    /// `true` if the scalars are contiguosly arranged in memory
    var isContiguous: Bool { return dataShape.isContiguous }
    /// `true` if the view projects padded data
    var isPadded: Bool { return padding != nil }
    /// is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied.
    /// Used primarily for debugging and unit testing
    var lastAccessMutatedView: Bool { return tensorArray.lastAccessMutatedView }
    /// the name of the view, which can optionally be set to aid in debugging
    var name: String { return tensorArray.name }
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
                  tensorArray: TensorArray(),
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
                  tensorArray: other.tensorArray,
                  viewDataOffset: other.viewDataOffset,
                  isShared: other.isShared,
                  scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// elementCount
    var elementCount: Int {
        return isPadded ? shape.padded(with: padding!).elementCount :
            shape.elementCount
    }
    
    //--------------------------------------------------------------------------
    /// elementCount
    func getPadding(for dim: Int) -> Padding {
        return padding?[dim % padding!.count] ?? Padding(0)
    }
    
    //--------------------------------------------------------------------------
    /// sequence2ScalarArray
    static func sequence2ScalarArray<Seq>(_ sequence: Seq) -> [Scalar] where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Scalar: AnyConvertable
    {
        return sequence.map { Scalar(any: $0) }
    }

    //--------------------------------------------------------------------------
    /// initTensorArray
    /// a helper to correctly initialize the tensorArray object
    mutating func initTensorArray(_ tensorData: TensorArray?,
                                  _ name: String?, _ scalars: [Scalar]?) {
        if let tensorData = tensorData {
            tensorArray = tensorData
        } else {
            assert(shape.isContiguous, "new views should have a dense shape")
            let name = name ?? String(describing: Self.self)

            // allocate backing tensorArray
            if let scalars = scalars {
                assert(scalars.count == dataShape.elementCount,
                       "number of scalars does not match tensor extents")
                do {
                    tensorArray = try scalars.withUnsafeBufferPointer {
                        try TensorArray(copying: $0, name: name)
                    }
                } catch {
                    tensorArray = TensorArray()
                    _Streams.current.reportDevice(error: error)
                }
            } else {
                tensorArray = TensorArray(type: Scalar.self,
                                          count: dataShape.elementCount,
                                          name: name)
            }
        }
    }

    //--------------------------------------------------------------------------
    /// init<T>(shapedLike other:
    /// convenience initializer used by generics to create typed result
    /// views of a matching size.
    /// - Parameter other: the desired shape
    init<T>(shapedLike other: T,
            with extents: [Int]? = nil) where T: TensorView {
        
        // create dense shaped view with specified extents or matching other
        let newShape: DataShape
        if let extents = extents {
            newShape = DataShape(extents: extents)
        } else {
            newShape = other.shape.dense
        }
        
        self.init(shape: newShape,
                  dataShape: newShape,
                  name: nil,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
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
                  tensorArray: nil, viewDataOffset: 0,
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
                                tensorArray: tensorArray,
                                viewDataOffset: viewDataOffset,
                                isShared: isShared,
                                scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorArray
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorArray)
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorArray if read-write access
    /// will cause mutation
    ///
    /// NOTE: this must be called from inside the accessQueue.sync block
    private mutating func copyIfMutates(using stream: DeviceStream) throws {
        // for unit tests
        tensorArray.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        diagnostic("\(mutationString) \(name)(\(tensorArray.trackingId)) " +
            "\(String(describing: Scalar.self))[\(dataShape.elementCount)]",
            categories: [.dataCopy, .dataMutation])
        
        tensorArray = try TensorArray(type: Scalar.self,
                                      copying: tensorArray,
                                      using: stream)
        tensorArray.lastAccessMutatedView = true
    }
    
    //--------------------------------------------------------------------------
    /// waitForCompletion(on stream:
    /// if there is a pending write completion event from a different
    /// stream that has not occurred, then queue a wait for it on this stream
    ///
    /// NOTE: this must be called from inside the accessQueue.sync block
    private func waitForCompletion(on stream: DeviceStream) throws {
        if let event = tensorArray.writeCompletionEvent, !event.occurred {
            try stream.wait(for: event)
            
            diagnostic(
                "\(waitString) \(stream.device.name)_\(stream.name) " +
                    "will wait for \(name)(\(tensorArray.trackingId)) " +
                    "\(String(describing: Scalar.self))" +
                "[\(dataShape.elementCount)]",
                categories: .scheduling)
        }
    }
    
    //--------------------------------------------------------------------------
    /// readOnly(using stream:
    /// Returns a read only device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    func readOnly(using stream: DeviceStream? = nil) throws
        -> UnsafeBufferPointer<Scalar>
    {
        let deviceStream = stream ?? _Streams.hostStream
        if let lastError = deviceStream.lastError { throw lastError }

        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            tensorArray.lastAccessMutatedView = false

            // queue a wait for pending writes
            try waitForCompletion(on: deviceStream)

            // get the buffer
            let buffer = try tensorArray.readOnly(type: Scalar.self,
                                                  using: deviceStream)
            
            // if no stream is specified then wait for completion
            // which will sync for host access
            if let event = tensorArray.writeCompletionEvent, stream == nil {
                event.wait()
            }

            return UnsafeBufferPointer(
                start: buffer.baseAddress!.advanced(by: viewDataOffset),
                count: dataShape.elementSpanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// readWrite(using stream:
    /// Returns a read write device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    mutating func readWrite(using stream: DeviceStream? = nil) throws
        -> UnsafeMutableBufferPointer<Scalar>
    {
        precondition(!tensorArray.isReadOnly, "the tensor is read only")
        let deviceStream = stream ?? _Streams.hostStream
        if let lastError = deviceStream.lastError { throw lastError }

        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            // queue a wait for pending writes
            try waitForCompletion(on: deviceStream)

            // mutating write?
            try copyIfMutates(using: deviceStream)
            
            // get the buffer
            let buffer = try tensorArray.readWrite(type: Scalar.self,
                                                   using: deviceStream)
            
            // if no stream is specified then wait for completion
            // which will sync for host access
            if let event = tensorArray.writeCompletionEvent, stream == nil {
                event.wait()
            }
            
            return UnsafeMutableBufferPointer(
                start: buffer.baseAddress!.advanced(by: viewDataOffset),
                count: dataShape.elementSpanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// createAndSetCompletionEvent
    /// creates a new completion event and sets it on the tensor
    func createAndSetCompletionEvent(using stream: DeviceStream) throws
        -> StreamEvent
    {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            // create a new event
            let event = try stream.createEvent()
            tensorArray.writeCompletionEvent = event
            
            diagnostic("\(stream.device.name)_\(stream.name) will signal " +
                "StreamEvent(\(event.trackingId))" +
                " when \(name)(\(tensorArray.trackingId)) " +
                "\(String(describing: Scalar.self))" +
                "[\(dataShape.elementCount)] is complete",
                categories: .scheduling)
            return event
        }
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorArray relative to this view
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
            tensorArray: tensorArray,
            viewDataOffset: subViewOffset,
            isShared: isReference,
            scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
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
                         tensorArray: tensorArray,
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
    mutating func reference(using stream: DeviceStream) throws -> Self {
        // get the queue, if we reference it as a tensorArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self.init(shape: shape,
                             dataShape: dataShape,
                             name: name,
                             padding: padding,
                             padValue: padValue,
                             tensorArray: tensorArray,
                             viewDataOffset: viewDataOffset,
                             isShared: true,
                             scalars: nil)
        }
    }
    
    //--------------------------------------------------------------------------
    /// referenceView
    /// Creates a reference view relative to this view. Write operations will
    /// not cause mutation of tensorArray. It's purpose is to support
    /// multi-threaded write operations
    
    // TODO: maybe remove this if a subview view can correctly be taken
    // from a `reference` view
    mutating func referenceView(
        offset: [Int], extents: [Int],
        padding: [Padding]? = nil, padValue: Scalar? = nil,
        using stream: DeviceStream) throws -> Self {
        
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
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
    /// Write operations will not cause mutation of tensorArray.
    /// It's purpose is to support multi-threaded write operations

    // TODO: maybe remove this if a subview view can correctly be taken
    // from a `reference` view
    mutating func referenceFlattened(
        axis: Int = 0,
        padding: [Padding]? = nil, padValue: Scalar? = nil,
        using stream: DeviceStream) throws -> Self {
        
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createFlattened(axis: axis, isShared: true,
                                   padding: padding, padValue: padValue)
        }
    }
}

//==============================================================================
//
public extension TensorView where Scalar: FloatingPoint {
    //--------------------------------------------------------------------------
    /// isFinite
    /// `true` if all elements are finite values. Primarily used for debugging
    func isFinite() throws -> Bool {
        let values = try readOnly()
        for value in values {
            if !value.isFinite {
                return false
            }
        }
        return true
    }
}

//==============================================================================
// map
public extension Zip2Sequence {
    typealias Pair = (Sequence1.Element, Sequence2.Element)
    
    /// map tensors
    @inlinable
    func map<T: TensorView>(to result: inout T,
                            _ transform: (Pair) -> T.Scalar) throws
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let pair = iterator.next() {
                results[i] = transform(pair)
            }
        }
    }

    /// map to a mutable collection
    @inlinable
    func map<Result: MutableCollection>(to result: inout Result,
                                        _ transform: (Pair) -> Result.Element) {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let pair = iterator.next() {
                result[i] = transform(pair)
            }
        }
    }
}

public extension Sequence {
    /// map a sequence to a tensor
    @inlinable
    func map<T: TensorView>(to result: inout T,
                            _ transform: (Element) -> T.Scalar) throws
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let value = iterator.next() {
                results[i] = transform(value)
            }
        }
    }

    /// map to a mutable collection
    @inlinable
    func map<Result: MutableCollection>(
        to result: inout Result, _ transform: (Element) -> Result.Element) {

        var iterator = self.makeIterator()
        for i in result.indices {
            if let value = iterator.next() {
                result[i] = transform(value)
            }
        }
    }
}

//==============================================================================
// zip
public func zip<T1, T2>(_ t1: T1, _ t2: T2) throws ->
    Zip2Sequence<TensorValueCollection<T1>, TensorValueCollection<T2>>
    where T1: TensorView, T2: TensorView
{
    return try zip(t1.values(), t2.values())
}

//==============================================================================
// reduce
public extension Sequence {
    /// reduce to a tensor
    func reduce<T>(
        to result: inout T,
        _ initialResult: T.Scalar,
        _ nextPartialResult: (T.Scalar, T.Scalar) throws -> T.Scalar) throws
        where T: TensorView, T.Scalar == Self.Element
    {
        var results = try result.mutableValues()
        var partial = initialResult
        for value in self {
            partial = try nextPartialResult(partial, value)
        }
        results[results.startIndex] = partial
    }

    /// reduce to a mutable collection
    @inlinable
    func reduce<Result: MutableCollection>(
        to result: inout Result,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) throws -> Result.Element)
        rethrows where Self.Element == Result.Element
    {
        var partial = initialResult
        for value in self {
            partial = try nextPartialResult(partial, value)
        }
        result[result.startIndex] = partial
    }
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
