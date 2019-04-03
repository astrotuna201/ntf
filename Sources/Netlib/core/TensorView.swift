//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorView
public protocol TensorView: AnyScalar, Logging, Equatable {
    /// The type of scalar referenced by the view
    associatedtype Scalar: AnyScalar
    /// A concrete type used in generics to return Boolean results
    associatedtype BoolView: TensorView
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView
    
    /// `true` if the scalars are densely packed in memory
    var isContiguous: Bool { get }
    /// `true` if the view contains zero elements
    var isEmpty: Bool { get }
    /// lastAccessMutated is `true` if the last data access caused the view
    /// to mutate, which causes the underlying tensorData object to be copied
    /// It's primary use is in debugging and unit testing
    var lastAccessMutated: Bool { get }
    /// the logging information for the view
    var logging: LogInfo? { get set }
    /// the name of the view, which can optionally be set to aid in debugging
    var name: String { get set }
    /// the number of dimensions in the view
    var rank: Int { get }
    /// the shape of the view
    var shape: DataShape { get }
    
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
    /// from a scalar in op functions.
    init(asScalar value: Scalar)
}

public extension TensorView {
    var isContiguous: Bool { return shape.isContiguous }
    var isEmpty: Bool { return shape.isEmpty }
    var rank: Int { return shape.rank }

    //--------------------------------------------------------------------------
    // helper to flip highlight color back and forth
    // it doesn't work in the Xcode console for some reason,
    // but it's nice when using CLion
    func setStringColor(text: inout String, highlight: Bool,
                        currentColor: inout LogColor,
                        normalColor: LogColor = .white,
                        highlightColor: LogColor = .blue) {
        #if os(Linux)
        if currentColor == normalColor && highlight {
            text += highlightColor.rawValue
            currentColor = highlightColor
            
        } else if currentColor == highlightColor && !highlight {
            text += normalColor.rawValue
            currentColor = normalColor
        }
        #endif
    }
}

public extension TensorView where Self: TensorViewImpl {
    //--------------------------------------------------------------------------
    /// init<T>(shapedLike other:
    /// convenience initializer used by generics
    init<T>(shapedLike other: T) where T: TensorView {
        self.init(shape: other.shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: nil,
                  logging: other.logging)
    }
    
    //--------------------------------------------------------------------------
    /// init<S>(asScalar value:
    /// convenience initializer used by generics
    init(asScalar value: Scalar) {
        // create scalar version of the shaped view type
        self.init(shape: DataShape(extents: [1]),
                  tensorData: nil, viewOffset: 0,
                  isShared: false, name: nil, logging: nil)
        // set the value
        try! rw()[0] = value
    }

    //--------------------------------------------------------------------------
    // name
    var name: String {
        get { return _name ?? String(describing: self) }
        set {
            _name = newValue
            if !tensorData.hasName { tensorData.name = newValue }
        }
    }
    
    //--------------------------------------------------------------------------
    // scalarValue
    func scalarValue() throws -> Scalar {
        assert(shape.elementCount == 1)
        return try ro()[0]
    }
}

//==============================================================================
// TensorViewImpl
public protocol TensorViewImpl: TensorView {
    /// `true` if this view is a reference view and does not cause mutation
    /// during write access. Primarily to support multi-threaded writes
    var isShared: Bool { get set }
    var _isShared: Bool { get set }
    /// optional name for the view
    var _name: String? { get set }
    /// class reference to the underlying byte buffer
    var tensorData: TensorData { get set }
    /// the linear element offset where the view begins
    var viewOffset: Int { get set }
    /// the linear byte offset where the view begins
    var viewByteOffset: Int { get }
    /// the number of bytes spanned by the view
    var viewSpanByteCount: Int { get }

    /// restated from TensorView with broader access control
    var lastAccessMutated: Bool { get set }
    var logging: LogInfo? { get set }
    var shape: DataShape { get set }

    /// determines if the view holds a unique reference to the underlying
    /// TensorData array
    mutating func isUniqueReference() -> Bool
}

//==============================================================================
// TensorViewImpl extension
//
public extension TensorViewImpl {
    //--------------------------------------------------------------------------
    // shared memory
    var isShared: Bool {
        get { return _isShared }
        set {
            assert(!newValue || isShared || isUniqueReference(),
                   "to set memory to shared it must already be shared or unique")
            _isShared = newValue
        }
    }

    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorData
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }
    
    //--------------------------------------------------------------------------
    /// viewByteOffset
    var viewByteOffset: Int { return viewOffset * MemoryLayout<Scalar>.size }
    
    //--------------------------------------------------------------------------
    // viewSpanByteCount
    var viewSpanByteCount: Int {
        return shape.elementSpanCount * MemoryLayout<Scalar>.size
    }

    //--------------------------------------------------------------------------
    /// isFinite
    /// `true` if all elements are finite values. Primarily used for debugging
    func isFinite() throws -> Bool {
        var isfiniteValue = true
        func check<T: AnyNumeric>(_ type: T.Type) throws {
            try ro().withMemoryRebound(to: AnyNumeric.self) {
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
        if lhs.tensorData === rhs.tensorData {
            // If they both reference the same tensorData then compare the views
            return lhs.viewOffset == rhs.viewOffset && lhs.shape == rhs.shape
            
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
    private mutating func copyIfMutates(using stream: DeviceStream? = nil) throws {
        // for unit tests
        lastAccessMutated = false
        guard !isShared && !isUniqueReference() else { return }
        
        lastAccessMutated = true
        if willLog(level: .diagnostic) == true {
            diagnostic("""
                \(mutationString) \(logging?.namePath ?? "")
                (\(tensorData.trackingId))  elements: \(shape.elementCount)
                """, categories: [.dataCopy, .dataMutation])
        }
        
        tensorData = try TensorData(withContentsOf: tensorData, using: stream)
    }
    
    //--------------------------------------------------------------------------
    /// ro
    /// Returns a read only tensorData buffer pointer synced with the
    /// applicaiton thread. It's purpose is to be used by shaped subscript
    /// functions
    func ro() throws -> UnsafeBufferPointer<Scalar> {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            let buffer = try tensorData.roHostRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// ro(using stream:
    /// Returns a read only device memory pointer synced with the specified
    /// stream. This version is by accelerator APIs
    func ro(using stream: DeviceStream) throws -> UnsafeRawPointer {
        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            let buffer = try tensorData.roDevicePointer(using: stream)
            return buffer.advanced(by: viewOffset)
        }
    }
    
    //--------------------------------------------------------------------------
    /// rw
    /// Returns a read write tensorData buffer pointer synced with the
    /// applicaiton thread. It's purpose is to be used by shaped subscript
    /// functions
    mutating func rw() throws -> UnsafeMutableBufferPointer<Scalar> {
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorData.accessQueue
        return try queue.sync {
            try copyIfMutates()
            let buffer = try tensorData.rwHostMutableRawBuffer()
            return buffer.bindMemory(to: Scalar.self)
        }
    }
    
    /// rw(using stream:
    /// Returns a read write device memory pointer synced with the specified
    /// stream. This version is by accelerator APIs
    mutating func rw(using stream: DeviceStream) throws -> UnsafeMutableRawPointer {
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
        let elementOffset = viewOffset + shape.linearIndex(of: offset)
        let subViewShape = DataShape(extents: extents,
                                     layout: shape.layout,
                                     channelLayout: shape.channelLayout,
                                     strides: shape.strides,
                                     isColMajor: shape.isColMajor)
        
        return Self.init(shape: subViewShape,
                         tensorData: tensorData,
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
                         tensorData: tensorData, viewOffset: viewOffset,
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
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self.init(shape: shape,
                             tensorData: tensorData, viewOffset: viewOffset,
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
        let queue = tensorData.accessQueue
        
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
        let queue = tensorData.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return createFlattened(axis: axis, isShared: true)
        }
    }
}
