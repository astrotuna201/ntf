//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// shaped positions and extents used for indexing and selection
public enum MatrixLayout { case rowMajor, columnMajor }
public typealias NDPosition = [Int]
public typealias ScalarPosition = Int
public typealias VectorPosition = Int
public typealias VectorExtents = Int
public typealias MatrixPosition = (r: Int, c: Int)
public typealias MatrixExtents = (rows: Int, cols: Int)
public typealias VolumePosition = (d: Int, r: Int, c: Int)
public typealias VolumeExtents = (depths: Int, rows: Int, cols: Int)
public typealias NCHWPosition = (i: Int, ch: Int, r: Int, c: Int)
public typealias NCHWExtents = (items: Int, channels: Int, rows: Int, cols: Int)
public typealias NHWCPosition = (i: Int, r: Int, c: Int, ch: Int)
public typealias NHWCExtents = (items: Int, rows: Int, cols: Int, channels: Int)

//==============================================================================
// ShapedTensorView
public protocol ShapedTensorView: TensorView {
    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         padding: [Padding]?,
         padValue: Element?,
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         isShared: Bool,
         values: [Values.Element]?)
}

public extension ShapedTensorView {
    //--------------------------------------------------------------------------
    /// DenseView
    func createDenseView(with extents: [Int]) -> Self {
        let shape = DataShape(extents: extents)
        return Self(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// DenseView
    func createDenseView(with value: Values.Element) -> Self {
        let extents = [Int](repeating: 1, count: rank)
        let shape = DataShape(extents: extents)
        return Self(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: [value])
    }
    
    //--------------------------------------------------------------------------
    /// repeated view
    init(with extents: [Int], repeating other: Self) {
        self.init(shape: DataShape(extents: extents),
                  dataShape: other.shape,
                  name: other.name,
                  padding: nil,
                  padValue: other.padValue,
                  tensorArray: other.tensorArray,
                  viewDataOffset: other.viewDataOffset,
                  isShared: other.isShared,
                  values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorArray relative to this view
    func createView(at offset: [Int], with extents: [Int],
                    isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))
        
        // the subview offset is the current plus the offset of index
        let subViewOffset = viewDataOffset + shape.linearIndex(of: offset)
        let subViewShape = DataShape(extents: extents, strides: shape.strides)
        let name = "\(self.name).subview"
        
        return Self(shape: subViewShape,
                    dataShape: subViewShape,
                    name: name,
                    padding: padding,
                    padValue: padValue,
                    tensorArray: tensorArray,
                    viewDataOffset: subViewOffset,
                    isShared: isReference,
                    values: nil)
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
                        padding: padding,
                        padValue: padValue,
                        tensorArray: tensorArray,
                        viewDataOffset: viewDataOffset,
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
        let flatShape = shape.flattened()
        return Self(shape: flatShape,
                    dataShape: flatShape,
                    name: name,
                    padding: padding,
                    padValue: padValue,
                    tensorArray: tensorArray,
                    viewDataOffset: viewDataOffset,
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
}

//==============================================================================
// Indexing
public extension ShapedTensorView {
    /// returns a collection of read only values
    func values(using stream: DeviceStream?) throws
        -> TensorValueCollection<Self>
    {
        let buffer = try readOnly(using: stream)
        return try TensorValueCollection(view: self, buffer: buffer)
    }
    
    /// returns a collection of read write values
    mutating func mutableValues(using stream: DeviceStream?) throws
        -> TensorMutableValueCollection<Self>
    {
        let buffer = try readWrite(using: stream)
        return try TensorMutableValueCollection(view: &self, buffer: buffer)
    }
}

//==============================================================================
// ScalarView
public protocol ScalarView: ShapedTensorView {}

public extension ScalarView {
    //--------------------------------------------------------------------------
    var endIndex: ScalarIndex {
        return ScalarIndex(view: self, at: 0)
    }
    
    var startIndex: ScalarIndex {
        return ScalarIndex(endOf: self)
    }

    //--------------------------------------------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> ScalarValue<Bool> {
        let shape = DataShape(extents: extents)
        return ScalarValue<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> ScalarValue<IndexElement> {
        let shape = DataShape(extents: extents)
        return ScalarValue<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// ScalarValue
public struct ScalarValue<Element>: ScalarView
where Element: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

extension ScalarValue: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView
public protocol VectorView: ShapedTensorView { }

extension Vector: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    var endIndex: VectorIndex {
        return VectorIndex(endOf: self)
    }
    
    var startIndex: VectorIndex {
        return VectorIndex(view: self, at: 0)
    }

    //--------------------------------------------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //-------------------------------------
    /// empty array
    init(count: Int, name: String? = nil) {
        let shape = DataShape(extents: [count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //-------------------------------------
    /// with Array
    init(name: String? = nil, values: [Values.Element]) {
        let shape = DataShape(extents: [values.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        let values = Self.sequence2ScalarArray(sequence)
        let shape = DataShape(extents: [values.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)
        
        // create shape considering column major
        let shape = DataShape(extents: [buffer.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Vector<Bool> {
        let shape = DataShape(extents: extents)
        return Vector<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Vector<IndexElement> {
        let shape = DataShape(extents: extents)
        return Vector<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// Vector
public struct Vector<Element>: VectorView
where Element: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
// MatrixView
public protocol MatrixView: ShapedTensorView {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    var endIndex: MatrixIndex {
        return MatrixIndex(endOf: self)
    }
    
    var startIndex: MatrixIndex {
        return MatrixIndex(view: self, at: (0, 0))
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: MatrixExtents, repeating other: Self) {
        let extents = [extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //-------------------------------------
    /// with Array
    init(_ extents: MatrixExtents,
         name: String? = nil,
         layout: MatrixLayout = .rowMajor,
         values: [Values.Element]? = nil)
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: MatrixExtents, name: String? = nil,
              layout: MatrixLayout = .rowMajor, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        self.init(extents, name: name, layout: layout,
                  values: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: MatrixExtents,
         layout: MatrixLayout = .rowMajor,
         referenceTo buffer: UnsafeBufferPointer<Element>,
         name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Matrix<Bool> {
        let shape = DataShape(extents: extents)
        return Matrix<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Matrix<IndexElement> {
        let shape = DataShape(extents: extents)
        return Matrix<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    // transpose
    var t: Self {
        return Self.init(shape: shape.transposed(),
                         dataShape: dataShape.transposed(),
                         name: name,
                         padding: padding,
                         padValue: padValue,
                         tensorArray: tensorArray,
                         viewDataOffset: viewDataOffset,
                         isShared: isShared,
                         values: nil)
    }
}


//==============================================================================
// Matrix
public struct Matrix<Element>: MatrixView where Element: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
// VolumeView
public protocol VolumeView: ShapedTensorView { }

extension Volume: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VolumeView extension
public extension VolumeView {
    //--------------------------------------------------------------------------
    var endIndex: VolumeIndex {
        return VolumeIndex(endOf: self)
    }

    var startIndex: VolumeIndex {
        return VolumeIndex(view: self, at: (0, 0, 0))
    }

    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: VolumeExtents, repeating other: Self) {
        
        let extents = [extents.depths, extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: VolumeExtents, name: String? = nil,
         values: [Values.Element]? = nil)
    {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: VolumeExtents, name: String? = nil,
              sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        self.init(extents, name: name,
                  values: Self.sequence2ScalarArray(sequence))
    }
    
    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: VolumeExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Volume<Bool> {
        let shape = DataShape(extents: extents)
        return Volume<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Volume<IndexElement> {
        let shape = DataShape(extents: extents)
        return Volume<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
/// Volume
public struct Volume<Element>: VolumeView
where Element: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: ShapedTensorView { }

extension NDTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// NDTensorView extensions
public extension NDTensorView {
    //--------------------------------------------------------------------------
    var endIndex: NDIndex {
        return NDIndex(endOf: self)
    }
    
    var startIndex: NDIndex {
        return NDIndex(view: self, at: [0])
    }

    //--------------------------------------------------------------------------
    /// with Sequence
    init<Seq>(extents: [Int], name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false,
                  values: Self.sequence2ScalarArray(sequence))
    }
    
    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(extents: [Int], name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NDTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NDTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NDTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NDTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Element>: NDTensorView
where Element: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
/// NCHWTensorView
/// An NCHW tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// c: channels
/// h: rows
/// w: cols
public protocol NCHWTensorView: ShapedTensorView { }

extension NCHWTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NCHWTensorView extensions
public extension NCHWTensorView {
    //--------------------------------------------------------------------------
    var endIndex: NDIndex {
        return NDIndex(endOf: self)
    }
    
    var startIndex: NDIndex {
        return NDIndex(view: self, at: [0])
    }

    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NCHWExtents, repeating other: Self) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        self.init(with: extent, repeating: other)
    }
    
    //-------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NCHWExtents, name: String? = nil,
         values: [Values.Element]? = nil) {

        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        let shape = DataShape(extents: extent)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NCHWExtents, name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        self.init(extents, name: name,
                  values: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NCHWExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NCHWTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NCHWTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NCHWTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NCHWTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
// NCHWTensor
public struct NCHWTensor<Element>: NCHWTensorView
where Element: DefaultInitializer {
    // types
    public typealias Element = Element
    
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
/// NHWCTensorView
/// An NHWC tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// h: rows
/// w: cols
/// c: channels
public protocol NHWCTensorView: ShapedTensorView { }

extension NHWCTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NHWCTensorView extensions
public extension NHWCTensorView {
    //--------------------------------------------------------------------------
    var endIndex: NDIndex {
        return NDIndex(endOf: self)
    }
    
    var startIndex: NDIndex {
        return NDIndex(view: self, at: [0])
    }

    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NHWCExtents, repeating other: Self) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        self.init(with: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with single value
    init(_ value: Values.Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NHWCExtents, name: String? = nil,
         values: [Values.Element]? = nil) {
        
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }

    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NHWCExtents, name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Values.Element: AnyConvertable
    {
        self.init(extents, name: name,
                  values: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NHWCExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NHWCTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NHWCTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NHWCTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NHWCTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            padding: nil, padValue: nil,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
/// NHWCTensor
public struct NHWCTensor<Element>: NHWCTensorView
where Element: DefaultInitializer {
    // types
    public typealias Element = Element
    
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}

//==============================================================================
/// NHWCTensor cast
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense uniform values to NHWC
    init<M>(_ matrix: M, name: String? = nil) where
        M: MatrixView,
        M.Element: UniformDenseScalar,
        M.Element.Component == Element {
            let viewExtents = [1,
                               matrix.shape.extents[0],
                               matrix.shape.extents[1],
                               M.Element.componentCount]
            let dataExtents = [1,
                               matrix.dataShape.extents[0],
                               matrix.dataShape.extents[1],
                               M.Element.componentCount]

            self.init(shape: DataShape(extents: viewExtents),
                      dataShape: DataShape(extents: dataExtents),
                      name: name,
                      padding: nil,
                      padValue: nil,
                      tensorArray: matrix.tensorArray,
                      viewDataOffset: matrix.viewDataOffset,
                      isShared: matrix.isShared,
                      values: nil)
    }
}
