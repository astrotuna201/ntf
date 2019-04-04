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
public protocol AnyUniformDenseScalar: AnyScalar {
    associatedtype ComponentScalar: AnyFixedSizeScalar
    static var componentCount: Int { get }
}

public extension AnyUniformDenseScalar {
    static var componentCount: Int {
        return MemoryLayout<Self>.size / MemoryLayout<ComponentScalar>.size
    }
}

//==============================================================================
// Image Scalar types
public protocol AnyRGBImageSample: AnyUniformDenseScalar {
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

public protocol AnyRGBAImageSample: AnyUniformDenseScalar {
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
public protocol AnyStereoAudioSample: AnyUniformDenseScalar {
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
public protocol ScalarTensorView: TensorView {
}

//------------------------------------------------------------------------------
// ScalarTensorViewImpl
public protocol ScalarTensorViewImpl: TensorViewImpl, ScalarTensorView where
    BoolView == ScalarTensor<Bool>,
    IndexView == ScalarTensor<TensorIndex>{ }

public extension ScalarTensorViewImpl {
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(_ value: Scalar, name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: [1], layout: .scalar)
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar: AnyScalar>: ScalarTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData

    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 0, "rank must equal: 0")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
}

//==============================================================================
// VectorTensorView
public protocol VectorTensorView: TensorView {
    var count: Int { get }
}

public extension VectorTensorView
where Self: VectorTensorViewImpl, Scalar: AnyConvertable {
//    //--------------------------------------------------------------------------
//    // formatted
//    func formatted(
//        precision: Int = 6,
//        columnWidth: Int = 9,
//        maxCols: Int = 10,
//        maxItems: Int = Int.max,
//        highlightThreshold: Float = Float.greatestFiniteMagnitude) -> String {
//
//        // setup
//        let itemCount = min(shape.items, maxItems)
//        let itemStride = shape.strides[0]
//        var string = "DataView extent \(shape.extents.description)\n"
//        var currentColor = LogColor.white
//        string += currentColor.rawValue
//
//        do {
//            let buffer = try ro()
//            string = "DataView extent [\(shape.items)]\n"
//            let pad = itemCount > 9 ? " " : ""
//            
//            for item in 0..<itemCount {
//                if item < 10 { string += pad }
//                string += "[\(item)] "
//                
//                let value = buffer[item * itemStride]
//                setStringColor(text: &string,
//                               highlight: value.asFloat > highlightThreshold,
//                               currentColor: &currentColor)
//                string += "\(String(format: format, value.asCVarArg))\n"
//            }
//            string += "\n"
//        } catch {
//            string += String(describing: error)
//        }
//        return string
//    }
}

//------------------------------------------------------------------------------
// VectorTensorViewImpl
public protocol VectorTensorViewImpl: TensorViewImpl, VectorTensorView
    where BoolView == VectorTensor<Bool>, IndexView == VectorTensor<TensorIndex>{}

public extension VectorTensorViewImpl {
    /// the number of elements in the vector
    var count: Int { return shape.extents[0] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(count: Int, name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: [count])
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
    
    /// initialize with scalar array
    init(scalars: [Scalar], name: String? = nil,
                logging: LogInfo? = nil) {
        self.init(count: scalars.count, name: name, logging: logging)
        _ = try! readWrite().initialize(from: scalars)
    }
}

//------------------------------------------------------------------------------
// VectorTensor
public struct VectorTensor<Scalar: AnyScalar>: VectorTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
    
    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 1, "rank must equal: 1")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
}

//==============================================================================
// MatrixTensorView
public protocol MatrixTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//------------------------------------------------------------------------------
// MatrixTensorViewImpl
public protocol MatrixTensorViewImpl: TensorViewImpl, MatrixTensorView
where BoolView == MatrixTensor<Bool>, IndexView == MatrixTensor<TensorIndex>{}

public extension MatrixTensorViewImpl {
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
    var isColMajor: Bool { return shape.isColMajor }
    
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int], scalars: [Scalar]? = nil, name: String? = nil,
         logging: LogInfo? = nil, isColMajor: Bool = false) {
        
        let shape = DataShape(extents: extents, isColMajor: isColMajor)
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ rows: Int, _ cols: Int, isColMajor: Bool = false,
         name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [rows, cols], name: name,
                  logging: logging, isColMajor: isColMajor)
    }
}

//------------------------------------------------------------------------------
// MatrixTensor
public struct MatrixTensor<Scalar: AnyScalar>: MatrixTensorViewImpl {
    // associated types
    public typealias BoolView = MatrixTensor<Bool>
    public typealias IndexView = MatrixTensor<TensorIndex>

    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
    
    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 2, "rank must equal: 2")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
}

//==============================================================================
// VolumeTensorView
public protocol VolumeTensorView: TensorView {
    var depthStride: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

//------------------------------------------------------------------------------
// VolumeTensorViewImpl
public protocol VolumeTensorViewImpl: TensorViewImpl, VolumeTensorView
where BoolView == VolumeTensor<Bool>, IndexView == VolumeTensor<TensorIndex>{}

public extension VolumeTensorViewImpl {
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: extents)
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    init(_ depths: Int, _ rows: Int, _ cols: Int,
         name: String? = nil, logging: LogInfo? = nil) {
        
        let shape = DataShape(extents: [depths, rows, cols])
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
}

//------------------------------------------------------------------------------
/// VolumeTensor
public struct VolumeTensor<Scalar: AnyScalar>: VolumeTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData

    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 3, "rank must equal: 3")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView {
}

//------------------------------------------------------------------------------
// NDTensorViewImpl
public protocol NDTensorViewImpl: TensorViewImpl, NDTensorView
where BoolView == NDTensor<Bool>, IndexView == NDTensor<TensorIndex>{}

public extension NDTensorViewImpl {
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar: AnyScalar>: NDTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
    
    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
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
public protocol NCHWTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//------------------------------------------------------------------------------
// NCHWTensorViewImpl
public protocol NCHWTensorViewImpl: TensorViewImpl, NCHWTensorView
where BoolView == NCHWTensor<Bool>, IndexView == NCHWTensor<TensorIndex>{}

public extension NCHWTensorViewImpl {
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: extents, layout: .nchw)
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    init(_ items: Int, _ channels: Int, _ rows: Int, _ cols: Int,
         name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [items, channels, rows, cols],
                  name: name, logging: logging)
    }
}

//------------------------------------------------------------------------------
// NCHWTensor
public struct NCHWTensor<Scalar: AnyScalar>: NCHWTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData

    
    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 4, "other rank must equal: 4")
        assert(shape.layout == .nchw, "other shape layout must be: .nchw")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
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
public protocol NHWCTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// NHWCTensorViewImpl
public protocol NHWCTensorViewImpl: TensorViewImpl, NHWCTensorView
where BoolView == NHWCTensor<Bool>, IndexView == NHWCTensor<TensorIndex>{}

public extension NHWCTensorViewImpl {
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: extents, layout: .nhwc)
        self.init(shape: shape, tensorData: nil, viewOffset: 0,
                  isShared: false, name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    init(_ items: Int, _ rows: Int, _ cols: Int, _ channels: Int,
         name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [items, rows, cols, channels],
                  name: name, logging: logging)
    }
}

//------------------------------------------------------------------------------
// NHWCTensor
public struct NHWCTensor<Scalar: AnyScalar>: NHWCTensorViewImpl {
    // properties
    public var _isShared: Bool = false
    public var _name: String?
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData
    
    //--------------------------------------------------------------------------
    // initializers
    /// fully specified to support generic init
    public init(shape: DataShape, tensorData: TensorData? = nil,
                viewOffset: Int = 0, isShared: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        assert(shape.rank == 4, "other rank must equal: 4")
        assert(shape.layout == .nchw, "other shape layout must be: .nchw")
        self.shape = shape
        self.logging = logging
        let spanCount = shape.elementSpanCount * MemoryLayout<Scalar>.size
        self.tensorData = tensorData ??
            TensorData(byteCount: spanCount, logging: logging, name: name)
        assert(viewByteOffset + spanCount <= self.tensorData.byteCount)
    }
}

//------------------------------------------------------------------------------
public extension NHWCTensor {
    /// zero copy cast of a matrix of dense uniform scalars to NHWC
    init<M: MatrixTensorViewImpl>(_ matrix: M, name: String? = nil) where
        M.Scalar: AnyUniformDenseScalar,
        M.Scalar.ComponentScalar == Scalar {
            let extents = [1, matrix.shape.extents[0],
                           matrix.shape.extents[1], M.Scalar.componentCount]
            self.shape = DataShape(extents: extents, layout: .nhwc)
            self.logging = matrix.logging
            self.tensorData = matrix.tensorData
    }
}
