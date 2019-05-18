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
public protocol ShapedTensorView: TensorView
{
}

public extension ShapedTensorView where Values.Element == Element {
    //--------------------------------------------------------------------------
    /// DenseView
    func createDenseView(with value: Values.Element) -> Self {
        let extents = [Int](repeating: 1, count: rank)
        let shape = DataShape(extents: extents)
        return Self(shape: shape, dataShape: shape, name: name,
                    tensorArray: nil, viewDataOffset: 0,
                    isShared: false, values: [value])
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> ScalarValue<Bool> {
        let shape = DataShape(extents: extents)
        return ScalarValue<Bool>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> ScalarValue<IndexElement> {
        let shape = DataShape(extents: extents)
        return ScalarValue<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// ScalarValue
public struct ScalarValue<Element>: ScalarView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //-------------------------------------
    /// empty array
    init(count: Int, name: String? = nil) {
        let shape = DataShape(extents: [count])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //-------------------------------------
    /// with Array
    init(name: String? = nil, values: [Element]) {
        let shape = DataShape(extents: [values.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Element: AnyConvertable
    {
        let values = Self.sequence2ScalarArray(sequence)
        let shape = DataShape(extents: [values.count])
        self.init(shape: shape, dataShape: shape, name: name,
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Vector<Bool> {
        let shape = DataShape(extents: extents)
        return Vector<Bool>(shape: shape, dataShape: shape, name: name,
                            tensorArray: nil, viewDataOffset: 0,
                            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Vector<IndexElement> {
        let shape = DataShape(extents: extents)
        return Vector<IndexElement>(shape: shape, dataShape: shape, name: name,
                                    tensorArray: nil, viewDataOffset: 0,
                                    isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// Vector
public struct Vector<Element>: VectorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }
    
    //-------------------------------------
    /// with Array
    init(_ extents: MatrixExtents,
         name: String? = nil,
         layout: MatrixLayout = .rowMajor,
         values: [Element]? = nil)
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: MatrixExtents, name: String? = nil,
              layout: MatrixLayout = .rowMajor, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Element: AnyConvertable
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Matrix<Bool> {
        let shape = DataShape(extents: extents)
        return Matrix<Bool>(shape: shape, dataShape: shape, name: name,
                            tensorArray: nil, viewDataOffset: 0,
                            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Matrix<IndexElement> {
        let shape = DataShape(extents: extents)
        return Matrix<IndexElement>(shape: shape, dataShape: shape, name: name,
                                    tensorArray: nil, viewDataOffset: 0,
                                    isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    // transpose
    var t: Self {
        return Self.init(shape: shape.transposed(),
                         dataShape: dataShape.transposed(),
                         name: name,
                         tensorArray: tensorArray,
                         viewDataOffset: viewDataOffset,
                         isShared: isShared,
                         values: nil)
    }
}


//==============================================================================
// Matrix
public struct Matrix<Element>: MatrixView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: VolumeExtents, name: String? = nil,
         values: [Element]? = nil)
    {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: VolumeExtents, name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable, Element: AnyConvertable
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Volume<Bool> {
        let shape = DataShape(extents: extents)
        return Volume<Bool>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Volume<IndexElement> {
        let shape = DataShape(extents: extents)
        return Volume<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
/// Volume
public struct Volume<Element>: VolumeView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
        Seq: Sequence, Seq.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NDTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NDTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NDTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NDTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Element>: NDTensorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NCHWExtents, name: String? = nil,
         values: [Element]? = nil) {

        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        let shape = DataShape(extents: extent)
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NCHWExtents, name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable, Element: AnyConvertable
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NCHWTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NCHWTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NCHWTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NCHWTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
// NCHWTensor
public struct NCHWTensor<Element>: NCHWTensorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
    init(_ value: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NHWCExtents, name: String? = nil,
         values: [Element]? = nil) {
        
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, values: values)
    }

    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NHWCExtents, name: String? = nil, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable, Element: AnyConvertable
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
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, values: nil)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NHWCTensor<Bool> {
        let shape = DataShape(extents: extents)
        return NHWCTensor<Bool>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NHWCTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        return NHWCTensor<IndexElement>(
            shape: shape, dataShape: shape, name: name,
            tensorArray: nil, viewDataOffset: 0,
            isShared: false, values: nil)
    }
}

//==============================================================================
/// NHWCTensor
public struct NHWCTensor<Element>: NHWCTensorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
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
                      tensorArray: matrix.tensorArray,
                      viewDataOffset: matrix.viewDataOffset,
                      isShared: matrix.isShared,
                      values: nil)
    }
}
