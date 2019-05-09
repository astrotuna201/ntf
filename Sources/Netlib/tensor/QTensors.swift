//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// Quantizing
public protocol Quantizing where Self: TensorView, Scalar: AnyNumeric {
    /// converter object
    associatedtype Converter: Quantizer
    /// parameters type
    associatedtype Param: Numeric
    /// the scalar type presented by the view
    associatedtype Viewed: AnyNumeric
    
    /// the bias to apply during conversion
    var bias: Param { get set }
    /// the scale to apply during conversion
    var scale: Param { get set }
    /// converts Scalar <--> ViewedScalar
    var converter: Converter { get }
}

//==============================================================================
/// extensions for quantized tensors
public extension TensorView where Self: Quantizing {
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order
    /// called to synchronize with the app thread
    @inlinable @inline(__always)
    func values() throws -> QTensorValueCollection<Self> {
        return try QTensorValueCollection(view: self, buffer: readOnly())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues() throws -> QTensorMutableValueCollection<Self>{
        return try QTensorMutableValueCollection(view: &self,
                                                 buffer: readWrite())
    }
    
    //--------------------------------------------------------------------------
    /// get a Sequence of read only values in spatial order as an array
    @inlinable @inline(__always)
    func array() throws -> [Viewed] {
        return try [Viewed](values())
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of read only values in spatial order
    @inlinable @inline(__always)
    func values(using stream: DeviceStream) throws
        -> QTensorValueCollection<Self>
    {
        return try QTensorValueCollection(
            view: self, buffer: readOnly(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// asynchronously get a Sequence of mutable values in spatial order
    @inlinable @inline(__always)
    mutating func mutableValues(using stream: DeviceStream) throws
        -> QTensorMutableValueCollection<Self>
    {
        return try QTensorMutableValueCollection(
            view: &self, buffer: readWrite(using: stream))
    }
    
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    @inlinable @inline(__always)
    func value(at position: ViewIndex.Position) throws -> Viewed {
//        let buffer = try readOnly()
//        let index = ViewIndex.init(view: self, at: position)
        return converter.convert(stored: 0)   //.convert(stored: index.isPad ? padValue : buffer[index.dataIndex])
    }
    
    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Viewed, at position: ViewIndex.Position) throws {
        let buffer = try readWrite()
        let index = ViewIndex.init(view: self, at: position)
        buffer[index.dataIndex] = converter.convert(viewed: value)
    }
}

//==============================================================================
/// Quantizer
public protocol Quantizer {
    associatedtype Stored: AnyNumeric
    associatedtype Viewed: AnyNumeric
    associatedtype Param: Numeric

    /// the bias to apply during conversion
    var bias: Param { get }
    /// the scale to apply during conversion
    var scale: Param  { get }
    /// converts from Scalar to ViewedScalar
    func convert(stored: Stored) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Stored
}

public struct QConverter<Stored, Viewed>: Quantizer where
    Stored: AnyNumeric,
    Viewed: AnyNumeric
{
    public var bias: Viewed
    public var scale: Viewed
    
    public init(bias: Viewed, scale: Viewed) {
        self.bias = bias
        self.scale = scale
    }

    /// converts from Scalar to ViewedScalar
    public func convert(stored: Stored) -> Viewed {
        fatalError()
    }
    /// converts from Scalar to ViewedScalar
    public func convert(viewed: Viewed) -> Stored {
        fatalError()
    }
}

//==============================================================================
/// QTensorValues
public protocol QTensorValues: RandomAccessCollection {
    associatedtype View: TensorView & Quantizing
    
    var view: View { get }
    var buffer: UnsafeBufferPointer<View.Scalar> { get }
    var startIndex: View.ViewIndex { get }
    var endIndex: View.ViewIndex { get }
    var count: Int { get }
    var padValue: View.Scalar { get }
}

public extension QTensorValues {
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    func index(before i: View.ViewIndex) -> View.ViewIndex {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    func index(after i: View.ViewIndex) -> View.ViewIndex {
        return i.increment()
    }

    @inlinable @inline(__always)
    subscript(index: View.ViewIndex) -> View.Viewed {
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return view.converter.convert(stored: stored)
    }
}

//==============================================================================
/// QTensorValueCollection
public struct QTensorValueCollection<View>: QTensorValues where
    View: TensorView & Quantizing
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Scalar>
    public let startIndex: View.ViewIndex
    public let endIndex: View.ViewIndex
    public let count: Int
    public let padValue: View.Scalar
    public var bias: View.Param
    public var scale: View.Param

    public init(view: View, buffer: UnsafeBufferPointer<View.Scalar>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
        bias = view.bias
        scale = view.scale
    }
}

//==============================================================================
/// QTensorValues
public protocol QTensorMutableValues: RandomAccessCollection, MutableCollection{
    associatedtype View: TensorView & Quantizing
    
    var view: View { get }
    var buffer: UnsafeMutableBufferPointer<View.Scalar> { get }
    var startIndex: View.ViewIndex { get }
    var endIndex: View.ViewIndex { get }
    var count: Int { get }
    var padValue: View.Scalar { get }
}

public extension QTensorMutableValues {
    //--------------------------------------------------------------------------
    // MutableCollection
    @inlinable @inline(__always)
    func index(before i: View.ViewIndex) -> View.ViewIndex {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    func index(after i: View.ViewIndex) -> View.ViewIndex {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    subscript(index: View.ViewIndex) -> View.Viewed {
        get {
            let stored = index.isPad ? padValue : buffer[index.dataIndex]
            return view.converter.convert(stored: stored)
        }
        set {
            buffer[index.dataIndex] = view.converter.convert(viewed: newValue)
        }
    }
}

//==============================================================================
/// QTensorMutableValueCollection
public struct QTensorMutableValueCollection<View>: QTensorMutableValues where
    View: TensorView & Quantizing
{
    // properties
    public let view: View
    public let buffer: UnsafeMutableBufferPointer<View.Scalar>
    public let startIndex: View.ViewIndex
    public let endIndex: View.ViewIndex
    public let count: Int
    public let padValue: View.Scalar
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Scalar>) throws {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
    }
}

//==============================================================================
// QMatrix
public struct QMatrix<Scalar, Viewed>: MatrixView, Quantizing
    where Scalar: AnyNumeric, Viewed: AnyNumeric
{
    public typealias Converter = QConverter<Scalar, Viewed>
    public typealias ConverterParam = Converter.Param
    
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    //
    public var bias: ConverterParam
    public var scale: ConverterParam
    public var converter: Converter
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                scalars: [Scalar]?) {
        
        assert(scalars == nil || scalars!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Scalar()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        self.bias = Viewed(exactly: 0)!
        self.scale = Viewed(exactly: 1)!
        self.converter = QConverter<Scalar, Viewed>(bias: self.bias, scale: self.scale)
        initTensorArray(tensorArray, name, scalars)
    }
}

