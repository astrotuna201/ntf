//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// Non numeric Scalar types
public protocol AnyChannelScalar: AnyScalar {
    associatedtype ChannelScalar
}

//==============================================================================
// Non numeric Scalar types
public protocol AnyRGB: AnyChannelScalar {
    var r: ChannelScalar { get set }
    var g: ChannelScalar { get set }
    var b: ChannelScalar { get set }
}

public struct RGB<T: AnyNumeric>: AnyRGB {
    public typealias ChannelScalar = T
    public var r, g, b: T
    public init() { r = T(); g = T(); b = T() }
}

public protocol AnyRGBA: AnyChannelScalar {
    var r: ChannelScalar { get set }
    var g: ChannelScalar { get set }
    var b: ChannelScalar { get set }
    var a: ChannelScalar { get set }
}

public struct RGBA<T: AnyNumeric>: AnyRGBA {
    public typealias ChannelScalar = T
    public var r, g, b, a: T
    public init() { r = T(); g = T(); b = T(); a = T() }
}

//==============================================================================
// TensorDataView
public protocol TensorDataView: AnyScalar {
    // types
    associatedtype Scalar: AnyScalar

    // properties
    var isContiguous: Bool { get }
    var isEmpty: Bool { get }
    var lastAccessMutated: Bool { get }
    var logging: LogInfo? { get set }
    var name: String { get set }
    var rank: Int { get }
}

public extension TensorDataView where Self: TensorDataViewImpl {
    var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }
    var isContiguous: Bool { return shape.isContiguous }
    var isEmpty: Bool { return shape.isEmpty }
    var rank: Int { return shape.rank }
}

//==============================================================================
// TensorDataViewImpl
public protocol TensorDataViewImpl: TensorDataView {
    // properties
    var isShared: Bool { get set }
    var lastAccessMutated: Bool { get set }
    var logging: LogInfo? { get set }
    var name: String { get set }
    var shape: DataShape { get set }
    var tensorData: TensorData<Scalar> { get set }
    var viewOffset: Int { get set }
    
    /// determines if the view holds a unique reference to the underlying
    /// TensorData array
    mutating func isUniqueReference() -> Bool
}

public extension TensorDataViewImpl {
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorData)
    }
    
    var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }
}

//==============================================================================
// ScalarTensorView
public protocol ScalarTensorView: TensorDataView {
}

//--------------------------------------------------------------------------
// ScalarTensorViewImpl
public protocol ScalarTensorViewImpl: TensorDataViewImpl, ScalarTensorView {}

public extension ScalarTensorViewImpl {
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar: AnyScalar>: ScalarTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(_ value: Scalar? = nil, name: String? = nil, logging: LogInfo? = nil) {
        self.shape = DataShape(extents: [1], layout: .scalar)
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
        // TODO
//        try! rw()[0] = Scalar(any: value)
    }

    // empty
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// VectorTensorView
public protocol VectorTensorView: TensorDataView {
    var count: Int { get }
}

//------------------------------------------------------------------------------
// VectorTensorViewImpl
public protocol VectorTensorViewImpl: TensorDataViewImpl, VectorTensorView {}

public extension VectorTensorViewImpl {
    var count: Int { return shape.extents[0] }
}

//------------------------------------------------------------------------------
// VectorTensor
public struct VectorTensor<Scalar: AnyScalar>: VectorTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(_ count: Int, name: String? = nil, logging: LogInfo? = nil) {
        self.shape = DataShape(extents: [count])
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
    }

    // empty
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// MatrixTensorView
public protocol MatrixTensorView: TensorDataView {
    var rows: Int { get }
    var cols: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// MatrixTensorViewImpl
public protocol MatrixTensorViewImpl: TensorDataViewImpl, MatrixTensorView {}

public extension MatrixTensorViewImpl {
    var rows: Int { return shape.extents[0] }
    var cols: Int { return shape.extents[1] }
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
    var isColMajor: Bool { return shape.isColMajor }
}

//--------------------------------------------------------------------------
// MatrixTensor
public struct MatrixTensor<Scalar: AnyScalar>: MatrixTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil,
                logging: LogInfo? = nil, isColMajor: Bool = false) {
        assert(extents.count == 2)
        self.shape = DataShape(extents: extents, isColMajor: isColMajor)
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
    }

    public init(_ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [rows, cols], name: name, logging: logging)
    }

    // empty
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// VolumeTensorView
public protocol VolumeTensorView: MatrixTensorView {
    var depths: Int { get }
    var depthStride: Int { get }
}

//------------------------------------------------------------------------------
// VolumeTensorViewImpl
public protocol VolumeTensorViewImpl: TensorDataViewImpl, VolumeTensorView {}

public extension VolumeTensorViewImpl {
    var depths: Int { return shape.extents[0] }
    var rows: Int { return shape.extents[1] }
    var cols: Int { return shape.extents[2] }
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2]  }
}

//------------------------------------------------------------------------------
// MatrixTensor
public struct VolumeTensor<Scalar: AnyScalar>: VolumeTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        assert(extents.count == 2)
        self.shape = DataShape(extents: extents)
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
    }
    
    public init(_ depths: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [depths, rows, cols], name: name, logging: logging)
    }

    // empty
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar: AnyScalar>: TensorDataViewImpl {
    // properties
    public var isShared: Bool
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(shape: DataShape,
                tensorData: TensorData<Scalar>? = nil,
                viewOffset: Int = 0,
                isShared: Bool = false,
                name: String? = nil,
                logging: LogInfo? = nil) {
        // assign
        self.isShared = isShared
        self.logging = logging
        self.shape = shape
        self.viewOffset = viewOffset
        self.tensorData = tensorData ?? TensorData(
            elementCount: shape.elementCount, logging: logging, name: name)
        
        assert(viewOffset + shape.elementCount <= self.tensorData.elementCount)
    }

    // empty
    public init() {
        isShared = false
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
        viewOffset = 0
    }
}

//==============================================================================
// NCHWTensorView
/// An NCHW tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// c: channels
/// h: rows
/// w: cols
public protocol NCHWTensorView: TensorDataView {
    var rows: Int { get }
    var cols: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// NCHWTensorViewImpl
public protocol NCHWTensorViewImpl: TensorDataViewImpl, NCHWTensorView {}

public extension NCHWTensorViewImpl {
    var items: Int { return shape.extents[0] }
    var channels: Int { return shape.extents[1] }
    var rows: Int { return shape.extents[2] }
    var cols: Int { return shape.extents[3] }
    
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1]  }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3]  }
}

//------------------------------------------------------------------------------
// NCHWTensor
public struct NCHWTensor<Scalar: AnyNumeric>: NCHWTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil,
                logging: LogInfo? = nil, isColMajor: Bool = false) {
        assert(extents.count == 4)
        self.shape = DataShape(extents: extents, layout: .nchw)
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
    }
    
    public init(_ items: Int, _ depths: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [items, depths, rows, cols],
                  name: name, logging: logging)
    }

    // empty
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}
