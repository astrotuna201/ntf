//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// ScalarView
public protocol ScalarView: TensorView where
    BoolView == ScalarValue<Bool>,
    IndexView == ScalarValue<IndexScalar>{}

public typealias ScalarPosition = Int

public extension ScalarView {
    //--------------------------------------------------------------------------
    var endIndex: ScalarIndex {
        return ScalarIndex(view: self, at: 0)
    }
    
    var startIndex: ScalarIndex {
        return ScalarIndex(endOf: self)
    }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(_ value: Stored, name: String? = nil) {
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }
}

//------------------------------------------------------------------------------
// ScalarValue
public struct ScalarValue<Stored>: ScalarView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
    }
}

extension ScalarValue: CustomStringConvertible where Stored: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VectorView
public protocol VectorView: TensorView
where BoolView == Vector<Bool>, IndexView == Vector<IndexScalar> { }

public typealias VectorPosition = Int
public typealias VectorExtents = Int

extension Vector: CustomStringConvertible where Stored: AnyConvertable {
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

    /// shaped initializers
    init(_ value: Stored, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }
    
    /// with Array
    init(count: Int, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
    
    /// with Sequence
    init<Seq>(count: Int, name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        self.init(name: name,
                  padding: padding, padValue: padValue,
                  scalars: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {
        
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)
        
        // create shape considering column major
        let shape = DataShape(extents: [buffer.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }

    //--------------------------------------------------------------------------
    /// initialize with scalar array
    init(name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         scalars: [Stored]) {
        let shape = DataShape(extents: [scalars.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }

    /// with Sequence
    init<Seq>(name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        let scalars = Self.sequence2ScalarArray(sequence)
        let shape = DataShape(extents: [scalars.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// Vector
public struct Vector<Stored>: VectorView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
    }
}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView
where BoolView == Matrix<Bool>, IndexView == Matrix<IndexScalar> {}

public typealias MatrixPosition = (r: Int, c: Int)
public typealias MatrixExtents = (rows: Int, cols: Int)

public enum MatrixLayout { case rowMajor, columnMajor }

extension Matrix: CustomStringConvertible where Stored: AnyConvertable {
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
    /// shaped initializers
    init(_ value: Stored, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }
    
    //-------------------------------------
    /// with Array
    init(_ extents: MatrixExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         layout: MatrixLayout = .rowMajor, scalars: [Stored]? = nil) {
        
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }
    
    //-------------------------------------
    /// repeating
    init(_ extents: MatrixExtents, repeating other: Self) {
        let extents = [extents.rows, extents.cols]
        self.init(extents: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: MatrixExtents, name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              layout: MatrixLayout = .rowMajor, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        self.init(extents, name: name,
                  padding: padding, padValue: padValue,
                  layout: layout,
                  scalars: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: MatrixExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         layout: MatrixLayout = .rowMajor,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {

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
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
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
                         scalars: nil)
    }
}


//==============================================================================
// Matrix
public struct Matrix<Stored>: MatrixView where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
    }
}

//==============================================================================
// VolumeView
public protocol VolumeView: TensorView
where BoolView == Volume<Bool>, IndexView == Volume<IndexScalar> { }

public typealias VolumePosition = (d: Int, r: Int, c: Int)
public typealias VolumeExtents = (depths: Int, rows: Int, cols: Int)

extension Volume: CustomStringConvertible where Stored: AnyConvertable {
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
    /// shaped initializers
    init(_ value: Stored, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: VolumeExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         scalars: [Stored]? = nil) {
        
        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }
    
    //-------------------------------------
    /// repeating
    init(_ extents: VolumeExtents, repeating other: Self) {
        
        let extents = [extents.depths, extents.rows, extents.cols]
        self.init(extents: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: VolumeExtents, name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        self.init(extents, name: name,
                  padding: padding, padValue: padValue,
                  scalars: Self.sequence2ScalarArray(sequence))
    }
    
    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: VolumeExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.depths, extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
}

//==============================================================================
/// Volume
public struct Volume<Stored>: VolumeView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView
where BoolView == NDTensor<Bool>, IndexView == NDTensor<IndexScalar> { }

public typealias NDPosition = [Int]

extension NDTensor: CustomStringConvertible where Stored: AnyConvertable {
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

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(extents: [Int], name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        // create shape considering column major
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(extents: [Int], name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false,
                  scalars: Self.sequence2ScalarArray(sequence))
    }
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Stored>: NDTensorView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
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
public protocol NCHWTensorView: TensorView
where BoolView == NCHWTensor<Bool>, IndexView == NCHWTensor<IndexScalar> { }

public typealias NCHWExtents = (items: Int, channels: Int, rows: Int, cols: Int)

extension NCHWTensor: CustomStringConvertible where Stored: AnyConvertable {
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
    /// shaped initializers
    init(_ value: Stored, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NCHWExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         scalars: [Stored]? = nil) {

        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        let shape = DataShape(extents: extent)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }
    
    //-------------------------------------
    /// repeating
    init(_ extents: NCHWExtents, repeating other: Self) {
        let extent = [extents.items, extents.channels,
                      extents.rows, extents.cols]
        self.init(extents: extent, repeating: other)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NCHWExtents, name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              isColMajor: Bool = false, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        self.init(extents, name: name,
                  padding: padding, padValue: padValue,
                  scalars: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NCHWExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.channels,
                       extents.rows, extents.cols]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
}

//==============================================================================
// NCHWTensor
public struct NCHWTensor<Stored>: NCHWTensorView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
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
public protocol NHWCTensorView: TensorView
where BoolView == NHWCTensor<Bool>, IndexView == NHWCTensor<IndexScalar> { }

public typealias NHWCExtents = (items: Int, rows: Int, cols: Int, channels: Int)

extension NHWCTensor: CustomStringConvertible where Stored: AnyConvertable {
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
    /// shaped initializers
    init(value: Stored, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil) {
        
        let shape = DataShape(extents: [1, 1, 1, 1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: [value])
    }

    //-------------------------------------
    /// with Array
    init(_ extents: NHWCExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         scalars: [Stored]? = nil) {
        
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false, scalars: scalars)
    }

    //-------------------------------------
    /// repeating
    init(_ extents: NHWCExtents, repeating other: Self) {
        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        self.init(extents: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with Sequence
    init<Seq>(_ extents: NHWCExtents, name: String? = nil,
              padding: [Padding]? = nil, padValue: Stored? = nil,
              isColMajor: Bool = false, sequence: Seq) where
        Seq: Sequence, Seq.Element: AnyConvertable,
        Stored: AnyConvertable
    {
        self.init(extents, name: name,
                  padding: padding, padValue: padValue,
                  scalars: Self.sequence2ScalarArray(sequence))
    }

    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: NHWCExtents, name: String? = nil,
         padding: [Padding]? = nil, padValue: Stored? = nil,
         referenceTo buffer: UnsafeBufferPointer<Stored>) {

        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let tensorArray = TensorArray(referenceTo: buffer, name: name)

        let extents = [extents.items, extents.rows,
                       extents.cols, extents.channels]
        let shape = DataShape(extents: extents)
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorArray: tensorArray, viewDataOffset: 0,
                  isShared: false, scalars: nil)
    }
}

//==============================================================================
/// NHWCTensor
public struct NHWCTensor<Stored>: NHWCTensorView
where Stored: DefaultInitializer {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Stored
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Stored?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Stored]?) {

        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Stored()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, scalars)
    }
}

//==============================================================================
/// NHWCTensor cast
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense uniform scalars to NHWC
    init<M>(_ matrix: M, name: String? = nil) where
        M: MatrixView,
        M.Stored: UniformDenseScalar,
        M.Stored.Component == Stored {
            let viewExtents = [1,
                               matrix.shape.extents[0],
                               matrix.shape.extents[1],
                               M.Stored.componentCount]
            let dataExtents = [1,
                               matrix.dataShape.extents[0],
                               matrix.dataShape.extents[1],
                               M.Stored.componentCount]

            self.init(shape: DataShape(extents: viewExtents),
                      dataShape: DataShape(extents: dataExtents),
                      name: name,
                      padding: nil,
                      padValue: nil,
                      tensorArray: matrix.tensorArray,
                      viewDataOffset: matrix.viewDataOffset,
                      isShared: matrix.isShared,
                      scalars: nil)
    }
}
