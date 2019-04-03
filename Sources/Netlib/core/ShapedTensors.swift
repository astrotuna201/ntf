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

public struct RGBSample<T>: AnyRGBImageSample
    where T: AnyNumeric & AnyFixedSizeScalar {
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

public struct RGBASample<T> : AnyRGBAImageSample
where T: AnyNumeric & AnyFixedSizeScalar {
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

public struct StereoSample<T>: AnyStereoAudioSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ChannelScalar = T
    public var left, right: T
    public init() { left = T(); right = T() }
}

//==============================================================================
// ScalarTensorView
public protocol ScalarTensorView: TensorView {
}

//--------------------------------------------------------------------------
// ScalarTensorViewImpl
public protocol ScalarTensorViewImpl: TensorViewImpl, ScalarTensorView { }

public extension ScalarTensorViewImpl {
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar: AnyScalar>: ScalarTensorViewImpl {
    // associated types
    public typealias BoolView = ScalarTensor<Bool>
    public typealias IndexView = ScalarTensor<TensorIndex>
    public typealias ScalarView = ScalarTensor<Scalar>

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
    
    /// shaped init
    public init(_ value: Scalar, name: String? = nil, logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents: [1], layout: .scalar),
                  name: name, logging: logging)
    }
    
    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
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
public protocol VectorTensorViewImpl: TensorViewImpl, VectorTensorView {}

public extension VectorTensorViewImpl {
    /// the number of elements in the vector
    var count: Int { return shape.extents[0] }
}

//------------------------------------------------------------------------------
// VectorTensor
public struct VectorTensor<Scalar: AnyScalar>: VectorTensorViewImpl {
    // associated types
    public typealias BoolView = VectorTensor<Bool>
    public typealias IndexView = VectorTensor<TensorIndex>

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
    
    /// shaped init
    public init(count: Int, name: String? = nil, logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents: [count]),
                  name: name, logging: logging)
    }
    
    /// initialize with scalar array
    public init(scalars: [Scalar], name: String? = nil,
                logging: LogInfo? = nil) {
        self.init(count: scalars.count, name: name, logging: logging)
        _ = try! rw().initialize(from: scalars)
    }

    /// creates an empty view
    public init() {
        logging = nil
        shape = DataShape()
        tensorData = TensorData()
    }
}

//==============================================================================
// MatrixTensorView
public protocol MatrixTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// MatrixTensorViewImpl
public protocol MatrixTensorViewImpl: TensorViewImpl, MatrixTensorView {}

public extension MatrixTensorViewImpl {
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
    var isColMajor: Bool { return shape.isColMajor }
}

//--------------------------------------------------------------------------
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

    /// shaped init
    public init(extents: [Int], scalars: [Scalar]? = nil, name: String? = nil,
                logging: LogInfo? = nil, isColMajor: Bool = false) {

        self.init(shape: DataShape(extents: extents, isColMajor: isColMajor),
                  name: name, logging: logging)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! rw().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    public init(_ rows: Int, _ cols: Int, isColMajor: Bool = false,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [rows, cols], name: name,
                  logging: logging, isColMajor: isColMajor)
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
public protocol VolumeTensorView: TensorView {
    var depthStride: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

//------------------------------------------------------------------------------
// VolumeTensorViewImpl
public protocol VolumeTensorViewImpl: TensorViewImpl, VolumeTensorView {}

public extension VolumeTensorViewImpl {
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2] }
}

//------------------------------------------------------------------------------
/// VolumeTensor
public struct VolumeTensor<Scalar: AnyScalar>: VolumeTensorViewImpl {
    // associated types
    public typealias BoolView = VolumeTensor<Bool>
    public typealias IndexView = VolumeTensor<TensorIndex>

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

    /// shaped init
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents: extents),
                  name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    public init(_ depths: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(shape: DataShape(extents: [depths, rows, cols]),
                  name: name, logging: logging)
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
public protocol NDTensorView: TensorView {
}

//------------------------------------------------------------------------------
// NDTensorViewImpl
public protocol NDTensorViewImpl: TensorViewImpl, NDTensorView {}

public extension NDTensorViewImpl {
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar: AnyScalar>: NDTensorViewImpl {
    // associated types
    public typealias BoolView = NDTensor<Bool>
    public typealias IndexView = NDTensor<TensorIndex>

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
public protocol NCHWTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// NCHWTensorViewImpl
public protocol NCHWTensorViewImpl: TensorViewImpl, NCHWTensorView {}

public extension NCHWTensorViewImpl {
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }
}

//------------------------------------------------------------------------------
// NCHWTensor
public struct NCHWTensor<Scalar: AnyScalar>: NCHWTensorViewImpl {
    // associated types
    public typealias BoolView = NCHWTensor<Bool>
    public typealias IndexView = NCHWTensor<TensorIndex>

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

    // shaped
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: extents, layout: .nchw)
        self.init(shape: shape, name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    public init(_ items: Int, _ channels: Int, _ rows: Int, _ cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [items, channels, rows, cols],
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
public protocol NHWCTensorView: TensorView {
    var rowStride: Int { get }
    var colStride: Int { get }
}

//--------------------------------------------------------------------------
// NHWCTensorViewImpl
public protocol NHWCTensorViewImpl: TensorViewImpl, NHWCTensorView {}

public extension NHWCTensorViewImpl {
    var itemStride: Int { return shape.strides[0] }
    var channelStride: Int { return shape.strides[1] }
    var rowStride: Int { return shape.strides[2] }
    var colStride: Int { return shape.strides[3] }
}

//------------------------------------------------------------------------------
// NHWCTensor
public struct NHWCTensor<Scalar: AnyScalar>: NHWCTensorViewImpl {
    // associated types
    public typealias BoolView = NHWCTensor<Bool>
    public typealias IndexView = NHWCTensor<TensorIndex>

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
    
    // shaped
    public init(extents: [Int], name: String? = nil, logging: LogInfo? = nil) {
        let shape = DataShape(extents: extents, layout: .nhwc)
        self.init(shape: shape, name: name, logging: logging)
    }
    
    /// initialize with explicit labels
    public init(_ items: Int, _ rows: Int, _ cols: Int, _ channels: Int,
                name: String? = nil, logging: LogInfo? = nil) {
        self.init(extents: [items, rows, cols, channels],
                  name: name, logging: logging)
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
    /// zero copy cast of a matrix of dense uniform scalars to NHWC
    init<M: MatrixTensorViewImpl>(_ matrix: M, name: String? = nil) where
        M.Scalar: AnyDenseChannelScalar,
        M.Scalar.ChannelScalar == Scalar {
            let extents = [1, matrix.shape.extents[0],
                           matrix.shape.extents[1], M.Scalar.channels]
            self.shape = DataShape(extents: extents, layout: .nhwc)
            self.logging = matrix.logging
            self.tensorData = matrix.tensorData
    }
}
