//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// non numeric scalar types.
/// For example: MatrixTensor<RGBASample<Float>> -> NHWCTensor<Float>
///
public protocol UniformDenseScalar: AnyScalar {
    associatedtype ComponentScalar: AnyFixedSizeScalar
    static var componentCount: Int { get }
}

public extension UniformDenseScalar {
    static var componentCount: Int {
        return MemoryLayout<Self>.size / MemoryLayout<ComponentScalar>.size
    }
}

//==============================================================================
// Image Scalar types
public protocol AnyRGBImageSample: UniformDenseScalar {
    var r: ComponentScalar { get set }
    var g: ComponentScalar { get set }
    var b: ComponentScalar { get set }
}

public struct RGBSample<T>: AnyRGBImageSample
    where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var r, g, b: T
    public init() { r = T(); g = T(); b = T() }
}

public protocol AnyRGBAImageSample: UniformDenseScalar {
    var r: ComponentScalar { get set }
    var g: ComponentScalar { get set }
    var b: ComponentScalar { get set }
    var a: ComponentScalar { get set }
}

public struct RGBASample<T> : AnyRGBAImageSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var r, g, b, a: T
    public init() { r = T(); g = T(); b = T(); a = T() }
}

//==============================================================================
// Audio sample types
public protocol AnyStereoAudioSample: UniformDenseScalar {
    var left: ComponentScalar { get set }
    var right: ComponentScalar { get set }
}

public struct StereoSample<T>: AnyStereoAudioSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var left, right: T
    public init() { left = T(); right = T() }
}

//==============================================================================
// ScalarTensorView
public protocol ScalarTensorView: TensorView
    where BoolView == ScalarTensor<Bool>, IndexView == ScalarTensor<TensorIndex>{}

public extension ScalarTensorView {
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(_ value: Scalar,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil) {
        
        let shape = DataShape(extents: [1])
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
    }
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar: AnyScalar>: ScalarTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
}

//==============================================================================
// VectorTensorView
public protocol VectorTensorView: TensorView
where BoolView == VectorTensor<Bool>, IndexView == VectorTensor<TensorIndex> {
    // properties
    var count: Int { get }
}

public extension VectorTensorView {
    /// the number of elements in the vector
    var count: Int { return shape.extents[0] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    /// create empty space
    init(count: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil) {
        
        let shape = DataShape(extents: [count])
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
    }
    
    /// initialize with scalar array
    init(name: String? = nil,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         logging: LogInfo? = nil, scalars: [Scalar]) {
        
        self.init(count: scalars.count,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
        _ = try! readWrite().initialize(from: scalars)
    }
}

//------------------------------------------------------------------------------
// VectorTensor
public struct VectorTensor<Scalar: AnyScalar>: VectorTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
}

//==============================================================================
// MatrixTensorView
public protocol MatrixTensorView: TensorView
where BoolView == MatrixTensor<Bool>, IndexView == MatrixTensor<TensorIndex> {
    // properties
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension MatrixTensorView {
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
    
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         isColMajor: Bool = false, scalars: [Scalar]? = nil) {
        
        let shape = DataShape(extents: extents)
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ rows: Int, _ cols: Int, isColMajor: Bool = false,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {
        self.init(extents: [rows, cols],
                  padding: padding, padValue: padValue,
                  name: name, logging: logging,
                  isColMajor: isColMajor, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// MatrixTensor
public struct MatrixTensor<Scalar: AnyScalar>: MatrixTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
}

//==============================================================================
// VolumeTensorView
public protocol VolumeTensorView: TensorView
where BoolView == VolumeTensor<Bool>, IndexView == VolumeTensor<TensorIndex> {
    var depthStride: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension VolumeTensorView {
    // properties
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {
        
        self.init(shape: DataShape(extents: extents),
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ depths: Int, _ rows: Int, _ cols: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {
        
        self.init(extents: [depths, rows, cols],
                  padding: padding, padValue: padValue,
                  name: name, logging: logging,
                  scalars: scalars)
    }
}

//------------------------------------------------------------------------------
/// VolumeTensor
public struct VolumeTensor<Scalar: AnyScalar>: VolumeTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView
where BoolView == NDTensor<Bool>, IndexView == NDTensor<TensorIndex> {
    
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar: AnyScalar>: NDTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
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
where BoolView == NCHWTensor<Bool>, IndexView == NCHWTensor<TensorIndex> {
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension NCHWTensorView {
    // properties
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {
        
        let shape = DataShape(extents: extents)
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ items: Int, _ channels: Int, _ rows: Int, _ cols: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {
        
        self.init(extents: [items, channels, rows, cols],
                  padding: padding, padValue: padValue,
                  name: name, logging: logging, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// NCHWTensor
public struct NCHWTensor<Scalar: AnyScalar>: NCHWTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
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
where BoolView == NHWCTensor<Bool>, IndexView == NHWCTensor<TensorIndex> {
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension NHWCTensorView {
    // properties
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {

        let shape = DataShape(extents: extents)
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ items: Int, _ rows: Int, _ cols: Int, _ channels: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil, logging: LogInfo? = nil,
         scalars: [Scalar]? = nil) {

        self.init(extents: [items, rows, cols, channels],
                  padding: padding, padValue: padValue,
                  name: name, logging: logging, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// NHWCTensor
public struct NHWCTensor<Scalar: AnyScalar>: NHWCTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessMutated: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var logging: LogInfo? = nil
    public var padding: [Padding]? = nil
    public var padValue: Scalar = Scalar()
    public init() {}
}

//------------------------------------------------------------------------------
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense uniform scalars to NHWC
    init<M: MatrixTensorView>(_ matrix: M, name: String? = nil) where
        M.Scalar: UniformDenseScalar,
        M.Scalar.ComponentScalar == Scalar {
            let extents = [1, matrix.shape.extents[0],
                           matrix.shape.extents[1], M.Scalar.componentCount]
            self.init(shape: DataShape(extents: extents),
                      tensorData: matrix._tensorData, viewOffset: 0,
                      isShared: false, name: nil, logging: matrix.logging)
    }
}
