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
public protocol ScalarTensorViewImpl: TensorDataViewImpl, ScalarTensorView { }

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

    //--------------------------------------------------------------------------
    // initializers
    public init(value: Scalar, name: String? = nil, logging: LogInfo? = nil) {
        self.shape = DataShape(extents: [1], layout: .scalar)
        self.logging = logging
        self.tensorData = TensorData(
            byteCount: shape.elementSpanCount * MemoryLayout<Scalar>.size,
            logging: logging, name: name)
        // it's being initialized in host memory so it can't fail
        try! rw()[0] = value
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 0, "other rank must equal: 0")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    // TODO: figure out how to move to extension TensorDataViewImpl and
    // remove all duplicates in this file
    /// creates an empty view
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
    /// the number of elements in the vector
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
    
    //--------------------------------------------------------------------------
    // initializers
    public init(count: Int, name: String? = nil, logging: LogInfo? = nil) {
        self.shape = DataShape(extents: [count])
        self.logging = logging
        self.tensorData = TensorData(
            byteCount: shape.elementSpanCount * MemoryLayout<Scalar>.size,
            logging: logging, name: name)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 1, "other rank must equal: 1")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
    
    /// initialize with scalar array
    public init(scalars: [Scalar], name: String? = nil, logging: LogInfo? = nil) {
        self.init(count: scalars.count, name: name, logging: logging)
        _ = try! rw().initialize(from: scalars)
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
    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], scalars: [Scalar]? = nil, name: String? = nil,
                logging: LogInfo? = nil, isColMajor: Bool = false) {
        assert(extents.count == 2)
        self.shape = DataShape(extents: extents, isColMajor: isColMajor)
        self.logging = logging
        self.tensorData = TensorData(
            byteCount: shape.elementSpanCount * MemoryLayout<Scalar>.size,
            logging: logging, name: name)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! rw().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    public init(_ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [rows, cols], name: name, logging: logging)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 2, "other rank must equal: 2")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// VolumeTensorView
public protocol VolumeTensorView: TensorDataView {
    var depths: Int { get }
    var rows: Int { get }
    var cols: Int { get }
    var depthStride: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
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
    var colStride: Int { return shape.strides[2] }
}

//------------------------------------------------------------------------------
/// VolumeTensor
public struct VolumeTensor<Scalar: AnyScalar>: VolumeTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData

    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        assert(extents.count == 2)
        self.shape = DataShape(extents: extents)
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: name)
    }
    
    /// initialize with explicit labels
    public init(_ depths: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [depths, rows, cols], name: name, logging: logging)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 3, "other rank must equal: 3")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorDataView {
}

//------------------------------------------------------------------------------
// NDTensorViewImpl
public protocol NDTensorViewImpl: TensorDataViewImpl, NDTensorView {}

public extension NDTensorViewImpl {
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar: AnyScalar>: NDTensorViewImpl {
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
    
    //--------------------------------------------------------------------------
    // initializers
    public init(shape: DataShape, name: String? = nil, logging: LogInfo? = nil) {
        // assign
        self.logging = logging
        self.shape = shape
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
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

    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        assert(extents.count == 4)
        self.shape = DataShape(extents: extents, layout: .nchw)
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: name)
    }
    
    /// initialize with explicit labels
    public init(_ items: Int, _ depths: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        
        self.init(extents: [items, depths, rows, cols],
                  name: name, logging: logging)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 4, "other rank must equal: 4")
        assert(other.shape.layout == .nchw, "other shape layout must be: .nchw")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
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
    
    //--------------------------------------------------------------------------
    // initializers
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        assert(extents.count == 4)
        self.shape = DataShape(extents: extents, layout: .nhwc)
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: name)
    }
    
    /// initialize with explicit labels
    public init(_ items: Int, _ depths: Int, _ rows: Int, _ cols: Int,
         name: String? = nil, logging: LogInfo? = nil) {
        
        self.init(extents: [items, depths, rows, cols],
                  name: name, logging: logging)
    }
    
    /// copy properties but not any data
    public init<T: TensorDataView>(shapedLike other: T) {
        assert(other.rank == 4, "other rank must equal: 4")
        assert(other.shape.layout == .nhwc, "other shape layout must be: .nhwc")
        self.shape = other.shape.dense
        self.logging = other.logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = TensorData(byteCount: spanCount,
                                     logging: logging, name: other.name)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//------------------------------------------------------------------------------
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense scalars to channel components
    init<M: MatrixTensorViewImpl>(_ matrix: M, name: String? = nil) where
        M.Scalar: AnyDenseChannelScalar,
        M.Scalar.ChannelScalar == Scalar {
            let extents = [1, matrix.rows, matrix.cols, M.Scalar.channels]
            self.shape = DataShape(extents: extents, layout: .nhwc)
            self.logging = matrix.logging
            self.tensorData = matrix.tensorData
    }
}
