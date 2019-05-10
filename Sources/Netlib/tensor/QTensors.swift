//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// Quantizing
public protocol Quantizing where Self: TensorView {
    /// parameters type
    associatedtype Stored
    /// the scalar type presented by the view
    associatedtype Viewed
    
    /// the bias to apply during conversion
    var bias: Float { get set }
    /// the scale to apply during conversion
    var scale: Float { get set }
    /// converts Stored <--> Viewed
    var quantizer: Quantizer<Stored, Viewed> { get }
}

//==============================================================================
/// extensions for quantized tensors
public extension TensorView where Self: Quantizing, Stored == Scalar {
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
        let buffer = try readOnly()
        let index = ViewIndex.init(view: self, at: position)
        let stored: Stored = index.isPad ? padValue : buffer[index.dataIndex]
        return quantizer.convert(stored: stored)
    }

    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Viewed, at position: ViewIndex.Position) throws {
        let buffer = try readWrite()
        let index = ViewIndex.init(view: self, at: position)
        buffer[index.dataIndex] = quantizer.convert(viewed: value)
    }
}

//==============================================================================
/// QTensorValues
public protocol QTensorValues: RandomAccessCollection
    where Index: TensorIndex
{
    associatedtype Stored
    associatedtype Viewed
    
    var buffer: UnsafeBufferPointer<Stored> { get }
    var startIndex: Index { get }
    var endIndex: Index { get }
    var count: Int { get }
    var padValue: Stored { get }
    var quantizer: Quantizer<Stored, Viewed> { get }
}

public extension QTensorValues {
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    func index(before i: Index) -> Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    func index(after i: Index) -> Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    subscript(index: Index) -> Viewed {
        let stored = index.isPad ? padValue : buffer[index.dataIndex]
        return quantizer.convert(stored: stored)
    }
}

//==============================================================================
/// QTensorValueCollection
public struct QTensorValueCollection<View>: QTensorValues
    where View: TensorView & Quantizing
{
    public typealias Stored = View.Scalar
    public typealias Viewed = View.Viewed
    public typealias Index  = View.ViewIndex
    public typealias Converter = Quantizer<Stored, Viewed>

    // properties
    public let buffer: UnsafeBufferPointer<Stored>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Stored
    public var quantizer: Converter

    public init(view: View, buffer: UnsafeBufferPointer<Stored>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
        quantizer = Converter(scale: view.scale, bias: view.bias)
    }
}

//==============================================================================
/// QTensorValues
public protocol QTensorMutableValues: RandomAccessCollection, MutableCollection
    where Index: TensorIndex
{
    associatedtype Stored
    associatedtype Viewed
    
    var buffer: UnsafeMutableBufferPointer<Stored> { get }
    var startIndex: Index { get }
    var endIndex: Index { get }
    var count: Int { get }
    var padValue: Stored { get }
    var quantizer: Quantizer<Stored, Viewed> { get }
}

public extension QTensorMutableValues {
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    func index(before i: Index) -> Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    func index(after i: Index) -> Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    subscript(index: Index) -> Viewed {
        get {
            let stored = index.isPad ? padValue : buffer[index.dataIndex]
            return quantizer.convert(stored: stored)
        }
        set {
            buffer[index.dataIndex] = quantizer.convert(viewed: newValue)
        }
    }
}

//==============================================================================
/// QTensorMutableValueCollection
public struct QTensorMutableValueCollection<View>: QTensorMutableValues
    where View: TensorView & Quantizing
{
    public typealias Stored = View.Scalar
    public typealias Viewed = View.Viewed
    public typealias Index  = View.ViewIndex
    public typealias Converter = Quantizer<Stored, Viewed>
    
    // properties
    public let buffer: UnsafeMutableBufferPointer<Stored>
    public let startIndex: Index
    public let endIndex: Index
    public let count: Int
    public let padValue: Stored
    public var quantizer: Converter
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<Stored>) throws {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.elementCount
        padValue = view.padValue
        quantizer = Converter(scale: view.scale, bias: view.bias)
    }
}

//==============================================================================
// QMatrix
public struct QMatrix<Stored, Viewed>: MatrixView, Quantizing where
    Stored: ScalarConformance
{
    public typealias Scalar = Stored
    public typealias Converter = Quantizer<Stored, Viewed>
    
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
    public var bias: Float
    public var scale: Float
    public var quantizer: Converter
    
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
        self.bias = 0
        self.scale = 1
        self.quantizer = Converter(scale: self.scale, bias: self.bias)
        initTensorArray(tensorArray, name, scalars)
    }
}


