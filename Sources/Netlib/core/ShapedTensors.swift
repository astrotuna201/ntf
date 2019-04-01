//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance assures that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// non numeric scalar types.
/// For example: MatrixTensor<RGBASample<Float>> -> NHWCTensor<Float>
///
public protocol AnyDenseChannelScalar: AnyScalar {
    associatedtype ChannelScalar: AnyFixedSizeScalar
    static var channels: Int { get }
}

public extension AnyDenseChannelScalar {
    static var channels: Int {
        return MemoryLayout<Self>.size / MemoryLayout<ChannelScalar>.size
    }
}

//==============================================================================
// Image Scalar types
public protocol AnyRGBImageSample: AnyDenseChannelScalar {
    var r: ChannelScalar { get set }
    var g: ChannelScalar { get set }
    var b: ChannelScalar { get set }
}

public struct RGBSample<T: AnyNumeric & AnyFixedSizeScalar>: AnyRGBImageSample {
    public typealias ChannelScalar = T
    public var r, g, b: T
    public init() { r = T(); g = T(); b = T() }
}

public protocol AnyRGBAImageSample: AnyDenseChannelScalar {
    var r: ChannelScalar { get set }
    var g: ChannelScalar { get set }
    var b: ChannelScalar { get set }
    var a: ChannelScalar { get set }
}

public struct RGBASample<T: AnyNumeric & AnyFixedSizeScalar>: AnyRGBAImageSample {
    public typealias ChannelScalar = T
    public var r, g, b, a: T
    public init() { r = T(); g = T(); b = T(); a = T() }
}

//==============================================================================
// Audio sample types
public protocol AnyStereoAudioSample: AnyDenseChannelScalar {
    var left: ChannelScalar { get set }
    var right: ChannelScalar { get set }
}

public struct StereoSample<T: AnyNumeric & AnyFixedSizeScalar>: AnyStereoAudioSample {
    public typealias ChannelScalar = T
    public var left, right: T
    public init() { left = T(); right = T() }
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
    
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

    
    //--------------------------------------------------------------------------
    // initializers
    public init(value: Scalar, name: String? = nil, logging: LogInfo? = nil) {
        self.shape = DataShape(extents: [1], layout: .scalar)
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                logging: logging, name: name)
        // it's being initialized in host memory so it can't fail
        try! rw()[0] = value
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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

    
    //--------------------------------------------------------------------------
    // initializers
    public init(shape: DataShape,
                tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
? = nil,
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
/// NCHWTensorView
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
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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

    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
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

//==============================================================================
/// NHWCTensorView
/// An NHWC tensor is a standard layout for use with cuDNN.
/// It has a layout of numerics organized as:
/// n: items
/// h: rows
/// w: cols
/// c: channels
public protocol NHWCTensorView: TensorDataView {
    var rows: Int { get }
    var cols: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// NHWCTensorViewImpl
public protocol NHWCTensorViewImpl: TensorDataViewImpl, NHWCTensorView {}

public extension NHWCTensorViewImpl {
    var items: Int { return shape.extents[0] }
    var channels: Int { return shape.extents[1] }
    var rows: Int { return shape.extents[2] }
    var cols: Int { return shape.extents[3] }
    
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }
}

//------------------------------------------------------------------------------
// NHWCTensor
public struct NHWCTensor<Scalar: AnyNumeric>: NHWCTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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

    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        assert(extents.count == 4)
        self.shape = DataShape(extents: extents, layout: .nhwc)
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

//public extension NHWCTensor {
//    /// zero copy cast of a matrix of dense scalars
//    init<M: MatrixTensorViewImpl>(_ matrix: M, name: String? = nil) where
//        M.Scalar: AnyDenseChannelScalar,
//        M.Scalar.ChannelScalar == Scalar {
//            let extents = [1, matrix.rows, matrix.cols, M.Scalar.channels]
//            self.shape = DataShape(extents: extents, layout: .nhwc)
//            self.logging = matrix.logging
//            self.tensorData = TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
    public var tensorData: TensorData
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
(matrix.tensorData)
//    }
//}
