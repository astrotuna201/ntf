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
/// Data repeating (broadcasting) is an instrinsic feature
///
public protocol TensorView: Logging {
    //--------------------------------------------------------------------------
    /// the type of element stored by the tensor
    associatedtype Element
    /// A tensor shape specific indexer used to calculate a data buffer
    /// index based on a view's spatial position
    associatedtype Index: TensorIndexing
    /// the type of read only elements collection
    associatedtype Values: RandomAccessCollection
    /// the type of read write elements collection
    associatedtype MutableValues: RandomAccessCollection & MutableCollection
        where MutableValues.Element == Values.Element
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView
        where BoolView.Element == Bool, BoolView.Values.Element == Bool
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView where
        IndexView.Element == IndexElement,
        IndexView.Values.Element == IndexElement

    //--------------------------------------------------------------------------
    // Readonly properties for the caller begin with an underscore. Accessor
    // functions with correct access are exposed as protocol extensions.
    // this gives full access to protocol default implementations.

    /// the shape of the actual underlying data. If `dataShape.extents` do not
    /// match `shape.extents` then the data is repeated (broadcast) across
    /// all dimensions during iteration and indexing.
    /// If `dataShape` is nil, then it equals `shape`
    var dataShape: DataShape { get }
    /// returns an index one past the end of the tensor used for collections
    var endIndex: Index { get }
    /// an adjustment to indexes to align properly when using repeated data
    var indexAlignment: [Int] { get }
    /// used internally when obtaining write access to manage
    /// multi-threaded writes without causing `tensorArray` copy on write.
    var isShared: Bool { get }
    /// the virtual shape of the view used for indexing
    /// if `shape` and `dataShape` are not equal, then `dataShape` is repeated
    var shape: DataShape { get }
    /// returns the first tensor index used for collections
    var startIndex: Index { get }
    /// class reference to the underlying byte buffer
    var tensorArray: TensorArray { get set }
    /// the indexing traversal procedure to use
    var traversal: TensorTraversal { get }
    /// the linear element offset where the view begins
    var viewDataOffset: Int { get set }

    //--------------------------------------------------------------------------
    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         indexAlignment: [Int]?,
         isShared: Bool,
         values: [Element]?)

    //--------------------------------------------------------------------------
    /// creates a new dense view where `Element` equals `Bool`
    /// with the specified extents
    func createBoolView(with extents: [Int]) -> BoolView
    /// creates a new dense view of the same type with the specified extents
    func createDenseView(with extents: [Int]) -> Self
    /// creates a new dense view of the same type with the specified value
    func createDenseView(with value: Values.Element) -> Self
    /// creates a new dense view where `Element` equals `IndexElement`
    /// with the specified extents and initial values
    func createIndexView(with extents: [Int]) -> IndexView

    //--------------------------------------------------------------------------
    // indexing
    /// returns a collection of viewed elements
    func values(using stream: DeviceStream?) throws -> Values

    /// returns a collection of mutable viewed elements
    mutating func mutableValues(using stream: DeviceStream?) throws
        -> MutableValues
}

//==============================================================================
/// IndexElement
/// The data type used for tensors that contain tensor spatial index values
public typealias IndexElement = Int32

//--------------------------------------------------------------------------
/// zeroIndexAlignment(_ shape:
@inlinable @inline(__always)
func zeroIndexAlignment(_ rank: Int) -> [Int] {
    return [Int](repeating: 0, count: rank)
}

//==============================================================================
/// traversal
public enum TensorTraversal: Int32 { case normal, repeated }

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    // public property accessors
    /// the extents of the view
    var extents: [Int] { return shape.extents }
    /// `true` if the values are contiguosly arranged in memory
    var isContiguous: Bool { return dataShape.isContiguous }
    /// is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied.
    /// Used primarily for debugging and unit testing
    var lastAccessMutatedView: Bool { return tensorArray.lastAccessMutatedView }
    /// the name of the view, which can optionally be set to aid in debugging
    var name: String { return tensorArray.name }
    /// the number of dimensions in the view
    var rank: Int { return shape.rank }
    
    //--------------------------------------------------------------------------
    /// creates a view of the same type and shape as `self` with `Element`
    /// equal to `Bool`
    func createBoolView() -> BoolView {
        return createBoolView(with: extents)
    }

    /// creates a view of the same type and shape as `self`
    func createDenseView() -> Self {
        return createDenseView(with: extents)
    }

    /// creates a view of the same shape as `self` with `Element`
    /// equal to `IndexElement`
    func createIndexView() -> IndexView {
        return createIndexView(with: extents)
    }

    //--------------------------------------------------------------------------
    /// empty
    init() {
        self.init(shape: DataShape(),
                  dataShape: DataShape(),
                  name: nil,
                  tensorArray: TensorArray(),
                  viewDataOffset: 0,
                  indexAlignment: [0],
                  isShared: false,
                  values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// DenseView
    func createDenseView(with extents: [Int]) -> Self {
        let shape = DataShape(extents: extents)
        return Self(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            indexAlignment: zeroIndexAlignment(shape.rank),
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// repeated view
    init(with extents: [Int], repeating other: Self) {
        self.init(shape: DataShape(extents: extents),
                  dataShape: other.shape,
                  name: other.name,
                  tensorArray: other.tensorArray,
                  viewDataOffset: other.viewDataOffset,
                  indexAlignment: zeroIndexAlignment(extents.count),
                  isShared: other.isShared,
                  values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorArray relative to this view
    private func createView(at offset: [Int], extents: [Int],
                            isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))
        let name = "\(self.name).subview"
        let viewShape = DataShape(extents: extents, strides: shape.strides)

        switch traversal {
        case .normal:
            // the subview offset is the current plus the offset of index
            let dataOffset = viewDataOffset + shape.linearIndex(of: offset)
            return Self(shape: viewShape,
                        dataShape: viewShape,
                        name: name,
                        tensorArray: tensorArray,
                        viewDataOffset: dataOffset,
                        indexAlignment: indexAlignment,
                        isShared: isReference,
                        values: nil)

        case .repeated:
            let alignment = zip(offset, dataShape.extents).map { $0 % $1 }
            return Self(shape: viewShape,
                        dataShape: dataShape,
                        name: name,
                        tensorArray: tensorArray,
                        viewDataOffset: 0,
                        indexAlignment: alignment,
                        isShared: isReference,
                        values: nil)
        }

    }
    
    //--------------------------------------------------------------------------
    /// reference
    /// creation of a reference is for the purpose of reshaped writes
    /// and multi-threaded writes to prevent mutation.
    /// The data will be copied before reference view creation if
    /// not uniquely held. Reference views will not perform
    /// copy-on-write when a write pointer is taken
    mutating func reference(using stream: DeviceStream) throws -> Self {
        // get the queue, if we reference it as a tensorArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self(shape: shape,
                        dataShape: dataShape,
                        name: name,
                        tensorArray: tensorArray,
                        viewDataOffset: viewDataOffset,
                        indexAlignment: indexAlignment,
                        isShared: true,
                        values: nil)
        }
    }
    
    //--------------------------------------------------------------------------
    /// flattened
    /// Returns a view with all dimensions higher than `axis` set to 1
    /// and the extent of `axis` adjusted to be the new total element count
    func flattened(axis: Int = 0) -> Self {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }
        
        // create flattened view
         // TODO: the flat alignment is wrong
        let flatShape = shape.flattened()
        return Self(shape: flatShape,
                    dataShape: flatShape,
                    name: name,
                    tensorArray: tensorArray,
                    viewDataOffset: viewDataOffset,
                    indexAlignment: zeroIndexAlignment(flatShape.rank),
                    isShared: isShared,
                    values: nil)
    }

    //--------------------------------------------------------------------------
    /// initTensorArray
    /// a helper to correctly initialize the tensorArray object
    mutating func initTensorArray(_ tensorData: TensorArray?,
                                  _ name: String?,
                                  _ values: [Element]?) {
        if let tensorData = tensorData {
            tensorArray = tensorData
        } else {
            assert(shape.isContiguous, "new views should have a dense shape")
            let name = name ?? String(describing: Self.self)
            
            // allocate backing tensorArray
            if let values = values {
                assert(values.count == dataShape.elementCount,
                       "number of values does not match tensor extents")
                do {
                    tensorArray = try values.withUnsafeBufferPointer {
                        try TensorArray(copying: $0, name: name)
                    }
                } catch {
                    tensorArray = TensorArray()
                    _Streams.current.reportDevice(error: error)
                }
            } else {
                tensorArray = TensorArray(type: Element.self,
                                          count: dataShape.elementCount,
                                          name: name)
            }
        }
    }

    //--------------------------------------------------------------------------
    /// realized
    /// create a dense view where the elements are coalesced
    /// and potentially type converted when working with qtensors
    /// if it is already of the correct form, then `self` is reaturned
    func realized() throws -> Self {
        if shape.isContiguous && shape == dataShape {
            return self
        } else {
            var result = createDenseView()
            Netlib.copy(view: self, result: &result)
            return result
        }
    }

    //--------------------------------------------------------------------------
    /// a collection of viewed elements
    @inlinable @inline(__always)
    func values(using stream: DeviceStream? = nil) throws -> Values {
        return try values(using: stream)
    }
    
    //--------------------------------------------------------------------------
    /// a collection of mutable viewed elements
    @inlinable @inline(__always)
    mutating func mutableValues(using stream: DeviceStream? = nil) throws
        -> MutableValues
    {
        return try mutableValues(using: stream)
    }
    
    //--------------------------------------------------------------------------
    /// an array of viewed elements
    @inlinable @inline(__always)
    func array() throws -> [Values.Element] {
        return [Values.Element](try values())
    }

    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    @inlinable @inline(__always)
    func value(at position: Index.Position) throws -> Element {
        let buffer = try readOnly()
        let index = Index(view: self, at: position)
        return buffer[index.dataIndex]
    }
    
    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Element, at position: Index.Position) throws {
        let buffer = try readWrite()
        let index = Index(view: self, at: position)
        buffer[index.dataIndex] = value
    }
    
    //--------------------------------------------------------------------------
    /// squeezed(axes:
    /// performs a rank reduction by removing dimensions with an extent of 1
    /// - Parameter axes: the axes to squeeze. `nil` implies all axes.
    /// - Returns: the new data shape
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    func squeezed(axes: [Int]? = nil) -> NDTensor<Element> {
        let squeezedShape = shape.squeezed(axes: axes)
        return NDTensor<Element>(shape: squeezedShape,
                                 dataShape: squeezedShape,
                                 name: name,
                                 tensorArray: tensorArray,
                                 viewDataOffset: viewDataOffset,
                                 indexAlignment: nil,
                                 isShared: isShared,
                                 values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// sequence2ScalarArray
    static func sequence2ScalarArray<Seq>(_ sequence: Seq) -> [Element]
        where Seq: Sequence, Seq.Element: AnyConvertable,
        Element: AnyConvertable
    {
        return sequence.map { Element(any: $0) }
    }

    //--------------------------------------------------------------------------
    /// scalarValue
    /// - Returns: the single value in the tensor as a scalar
    func scalarValue() throws -> Element {
        assert(shape.elementCount == 1)
        return try readOnly()[0]
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
    mutating func copyIfMutates(using stream: DeviceStream) throws {
        // for unit tests
        tensorArray.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        diagnostic("\(mutationString) \(name)(\(tensorArray.trackingId)) " +
            "\(String(describing: Element.self))[\(dataShape.elementCount)]",
            categories: [.dataCopy, .dataMutation])
        
        tensorArray = try TensorArray(type: Element.self,
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
                    "\(String(describing: Element.self))" +
                "[\(dataShape.elementCount)]",
                categories: .scheduling)
        }
    }
    
    //--------------------------------------------------------------------------
    /// readOnly(using stream:
    /// Returns a read only device memory pointer synced with the specified
    /// stream. This version is used by accelerator APIs
    func readOnly(using stream: DeviceStream? = nil) throws
        -> UnsafeBufferPointer<Element>
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
            let buffer = try tensorArray.readOnly(type: Element.self,
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
        -> UnsafeMutableBufferPointer<Element>
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
            let buffer = try tensorArray.readWrite(type: Element.self,
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
                "\(String(describing: Element.self))" +
                "[\(dataShape.elementCount)] is complete",
                categories: .scheduling)
            return event
        }
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
    func view(at offset: [Int], extents: [Int]) -> Self {
        // the view created will have the same isShared state as the parent
        return createView(at: offset, extents: extents, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// viewItems
    /// Returns a view along the first dimension spanning all the others.
    /// It is used to simplify accessing a set of training samples.
    /// The view created will have the same isShared state as the parent
    func viewItems(at offset: Int, count: Int) -> Self {
        let index, viewExtents: [Int]
        if rank == 1 {
            index = [offset]
            viewExtents = [count]
        } else {
            index = [offset] + [Int](repeating: 0, count: rank - 1)
            viewExtents = [count] + shape.extents.suffix(from: 1)
        }
        
        return createView(at: index, extents: viewExtents, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// view(item:
    func view(item: Int) -> Self {
        return viewItems(at: item, count: 1)
    }
}

//==============================================================================
//
public extension TensorView where Element: FloatingPoint {
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
public extension Sequence {
    /// map a sequence to a tensor
    @inlinable
    func map<R>(to result: inout R,
                _ transform: (Element) -> R.MutableValues.Element) throws where
        R: TensorView
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
    func map<R>(to result: inout R,
                _ transform: (Element) -> R.Element) where
        R: MutableCollection
    {
        
        var iterator = self.makeIterator()
        for i in result.indices {
            if let value = iterator.next() {
                result[i] = transform(value)
            }
        }
    }
}

//==============================================================================
public extension Zip2Sequence {
    typealias Pair = (Sequence1.Element, Sequence2.Element)
    
    /// map tensors
    @inlinable
    func map<T>(to result: inout T,
                _ transform: (Pair) -> T.MutableValues.Element) throws
        where T: TensorView
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
    func map<Result>(to result: inout Result,
                     _ transform: (Pair) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let pair = iterator.next() {
                result[i] = transform(pair)
            }
        }
    }
}

//==============================================================================
public extension Zip3Sequence {
    typealias Input = (S1.Element, S2.Element, S3.Element)
    
    /// map tensors
    @inlinable
    func map<T>(to result: inout T,
                _ transform: (Input) -> T.MutableValues.Element) throws
        where T: TensorView
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let input = iterator.next() {
                results[i] = transform(input)
            }
        }
        
    }
    
    /// map to a mutable collection
    @inlinable
    func map<Result>(to result: inout Result,
                     _ transform: (Input) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let input = iterator.next() {
                result[i] = transform(input)
            }
        }
    }
}

//==============================================================================
// zip
public func zip<T1, T2>(_ t1: T1, _ t2: T2) throws ->
    Zip2Sequence<T1.Values, T2.Values>
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
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) throws -> Element) throws
        where T: TensorView, Element == T.MutableValues.Element
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
    func reduce<Result>(
        to result: inout Result,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) throws -> Result.Element) rethrows
        where Result: MutableCollection, Element == Result.Element
    {
        var partial = initialResult
        for value in self {
            partial = try nextPartialResult(partial, value)
        }
        result[result.startIndex] = partial
    }
}
