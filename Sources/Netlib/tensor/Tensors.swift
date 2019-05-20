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
// TensorView extensions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// createValueView
    func createValueView(_ value: Values.Element, name: String? = nil) -> Self {
        let extents = [Int](repeating: 1, count: rank)
        let shape = DataShape(extents: extents)
        let array = try! TensorArray(copying: [value],
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        return Self(shape: shape, dataShape: shape,
                    tensorArray: array, viewDataOffset: 0,
                    indexAlignment: zeroAlignment(shape.rank),
                    traversal: .normal, isShared: false)
    }
}

public extension TensorView where Values.Element == Element {
    //--------------------------------------------------------------------------
    /// returns a collection of read only values
    func values(using stream: DeviceStream?) throws
        -> TensorValueCollection<Self>
    {
        let buffer = try readOnly(using: stream)
        return try TensorValueCollection(view: self, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
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
public protocol ScalarView: TensorView {}

public extension ScalarView {
    //--------------------------------------------------------------------------
    var startIndex: ScalarIndex { return ScalarIndex(endOf: self) }
    var endIndex: ScalarIndex { return ScalarIndex(view: self, at: 0) }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> ScalarValue<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self, count: shape.elementCount,
                                name: String(describing: Self.self))
        return ScalarValue<Bool>(
            shape: shape, dataShape: shape,
            tensorArray: array, viewDataOffset: 0,
            indexAlignment: zeroAlignment(shape.rank),
            traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> ScalarValue<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return ScalarValue<IndexElement>(
            shape: shape, dataShape: shape,
            tensorArray: array, viewDataOffset: 0,
            indexAlignment: zeroAlignment(shape.rank),
            traversal: .normal, isShared: false)
    }
}

public extension ScalarView {
    //--------------------------------------------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        let array = try! TensorArray(copying: [element],
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//------------------------------------------------------------------------------
// ScalarValue
public struct ScalarValue<Element>: ScalarView {
    // properties
    public let shape: DataShape
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
    }
}

extension ScalarValue: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView
public protocol VectorView: TensorView { }

extension Vector: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    var startIndex: VectorIndex { return VectorIndex(view: self, at: 0) }
    var endIndex: VectorIndex { return VectorIndex(endOf: self) }

    //-------------------------------------
    /// empty array
    init(count: Int, name: String? = nil) {
        let shape = DataShape(extents: [count])
        let array = TensorArray(type: Element.self, count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        self.init(shape: shape, dataShape: shape,
                    tensorArray: array, viewDataOffset: 0,
                    indexAlignment: zeroAlignment(shape.rank),
                    traversal: .normal, isShared: false)
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: [buffer.count])
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Vector<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Vector<Bool>(shape: shape, dataShape: shape,
                            tensorArray: array, viewDataOffset: 0,
                            indexAlignment: zeroAlignment(shape.rank),
                            traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Vector<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Vector<IndexElement>(shape: shape, dataShape: shape,
                            tensorArray: array, viewDataOffset: 0,
                            indexAlignment: zeroAlignment(shape.rank),
                            traversal: .normal, isShared: false)
    }
}

//==============================================================================
public extension VectorView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1])
        let array = try! TensorArray(copying: [element],
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(name: name, elements: any.map { Element(any: $0) })
    }

    //-------------------------------------
    /// with an element collection
    init<C>(name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let shape = DataShape(extents: [elements.count])
        let array = try! TensorArray(copying: elements,
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//==============================================================================
// Vector
public struct Vector<Element>: VectorView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
    }
}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    var startIndex: MatrixIndex { return MatrixIndex(view: self, at: (0, 0)) }
    var endIndex: MatrixIndex { return MatrixIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: MatrixExtents, name: String? = nil) {
        let shape = DataShape(extents: [extents.rows, extents.cols])
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: MatrixExtents, repeating other: Self) {
        let extents = [extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
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
        let array = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Matrix<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Matrix<Bool>(shape: shape, dataShape: shape,
                            tensorArray: array, viewDataOffset: 0,
                            indexAlignment: zeroAlignment(shape.rank),
                            traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Matrix<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Matrix<IndexElement>(shape: shape, dataShape: shape,
                                    tensorArray: array, viewDataOffset: 0,
                                    indexAlignment: zeroAlignment(shape.rank),
                                    traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    // transpose
    var t: Self {
        let tAlign = [indexAlignment[1], indexAlignment[0]]
        return Self.init(shape: shape.transposed(),
                         dataShape: dataShape.transposed(),
                         tensorArray: tensorArray,
                         viewDataOffset: viewDataOffset,
                         indexAlignment: tAlign,
                         traversal: traversal,
                         isShared: isShared)
    }
}

//==============================================================================
// MatrixView data initialization extensions
public extension MatrixView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        let array = try! TensorArray(copying: [element],
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, layout: layout,
                  elements: any.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()

        let array = try! TensorArray(copying: elements,
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)

        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//==============================================================================
// Matrix
public struct Matrix<Element>: MatrixView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
    }
}

//==============================================================================
// VolumeView
public protocol VolumeView: TensorView { }

extension Volume: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VolumeView extension
public extension VolumeView {
    var startIndex: VolumeIndex { return VolumeIndex(view: self, at: (0, 0, 0))}
    var endIndex: VolumeIndex { return VolumeIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: VolumeExtents, name: String? = nil) {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: VolumeExtents, repeating other: Self) {
        let extents = [extents.depths, extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: VolumeExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> Volume<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Volume<Bool>(shape: shape, dataShape: shape,
                            tensorArray: array, viewDataOffset: 0,
                            indexAlignment: zeroAlignment(shape.rank),
                            traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> Volume<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return Volume<IndexElement>(shape: shape, dataShape: shape,
                                    tensorArray: array, viewDataOffset: 0,
                                    indexAlignment: zeroAlignment(shape.rank),
                                    traversal: .normal, isShared: false)
    }
}

//==============================================================================
// VolumeView extension
public extension VolumeView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1])
        let name = name ?? String(describing: Self.self)
        let array = try! TensorArray(copying: [element], name: name,
                                     using: _Streams.hostStream)
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(_ extents: VolumeExtents, name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, elements: any.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: VolumeExtents, name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = try! TensorArray(copying: elements, name: name,
                                     using: _Streams.hostStream)
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//==============================================================================
/// Volume
public struct Volume<Element>: VolumeView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int

    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView { }

extension NDTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// NDTensorView extensions
public extension NDTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: [Int], name: String? = nil) {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(extents: [Int], name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NDTensor<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NDTensor<Bool>(shape: shape, dataShape: shape,
                              tensorArray: array, viewDataOffset: 0,
                              indexAlignment: zeroAlignment(shape.rank),
                              traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NDTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NDTensor<IndexElement>(shape: shape, dataShape: shape,
                                      tensorArray: array, viewDataOffset: 0,
                                      indexAlignment: zeroAlignment(shape.rank),
                                      traversal: .normal, isShared: false)
    }
}

//==============================================================================
// NDTensorView extensions
public extension NDTensorView {
    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(_ extents: [Int], name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, elements: any.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: [Int], name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let shape = DataShape(extents: extents)
        let array = try! TensorArray(copying: elements,
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Element>: NDTensorView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int

    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
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
public protocol NCHWTensorView: TensorView { }

extension NCHWTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NCHWTensorView extensions
public extension NCHWTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: NCHWExtents, name: String? = nil) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        let shape = DataShape(extents: extent)
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NCHWExtents, repeating other: Self) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        self.init(with: extent, repeating: other)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NCHWExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NCHWTensor<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NCHWTensor<Bool>(shape: shape, dataShape: shape,
                                tensorArray: array, viewDataOffset: 0,
                                indexAlignment: zeroAlignment(shape.rank),
                                traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NCHWTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NCHWTensor<IndexElement>(
            shape: shape, dataShape: shape,
            tensorArray: array, viewDataOffset: 0,
            indexAlignment: zeroAlignment(shape.rank),
            traversal: .normal, isShared: false)
    }
}

//==============================================================================
/// NCHWTensorView extensions for data initialization
public extension NCHWTensorView {
    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1, 1, 1])
        let name = name ?? String(describing: Self.self)
        let array = try! TensorArray(copying: [element], name: name,
                                     using: _Streams.hostStream)
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(_ extents: NCHWExtents, name: String? = nil, any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, elements: any.map { Element(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: NCHWExtents, name: String? = nil, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        let name = name ?? String(describing: Self.self)
        let array = try! TensorArray(copying: elements, name: name,
                                     using: _Streams.hostStream)
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
}

//==============================================================================
// NCHWTensor
public struct NCHWTensor<Element>: NCHWTensorView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int

    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
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
public protocol NHWCTensorView: TensorView { }

extension NHWCTensor: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
/// NHWCTensorView extensions
public extension NHWCTensorView {
    var startIndex: NDIndex { return NDIndex(view: self, at: [0]) }
    var endIndex: NDIndex { return NDIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: NHWCExtents, name: String? = nil) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount,
                                name: name ?? String(describing: Self.self))
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: NHWCExtents, repeating other: Self) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        self.init(with: extents, repeating: other)
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NHWCExtents, name: String? = nil,
         referenceTo buffer: UnsafeBufferPointer<Element>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")

        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
    }

    //--------------------------------------------------------------------------
    /// BoolView
    func createBoolView(with extents: [Int]) -> NHWCTensor<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: Bool.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NHWCTensor<Bool>(shape: shape, dataShape: shape,
                                tensorArray: array, viewDataOffset: 0,
                                indexAlignment: zeroAlignment(shape.rank),
                                traversal: .normal, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexView(with extents: [Int]) -> NHWCTensor<IndexElement> {
        let shape = DataShape(extents: extents)
        let array = TensorArray(type: IndexElement.self,
                                count: shape.elementCount,
                                name: String(describing: Self.self))
        return NHWCTensor<IndexElement>(
            shape: shape, dataShape: shape,
            tensorArray: array, viewDataOffset: 0,
            indexAlignment: zeroAlignment(shape.rank),
            traversal: .normal, isShared: false)
    }
}

//==============================================================================
/// NHWCTensor
public struct NHWCTensor<Element>: NHWCTensorView {
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int

    public init(shape: DataShape,
                dataShape: DataShape,
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
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
            let alignment = matrix.indexAlignment + [0]

            self.init(shape: DataShape(extents: viewExtents),
                      dataShape: DataShape(extents: dataExtents),
                      tensorArray: matrix.tensorArray,
                      viewDataOffset: matrix.viewDataOffset,
                      indexAlignment: alignment,
                      traversal: matrix.traversal,
                      isShared: matrix.isShared)
    }
}
