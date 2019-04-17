//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// non numeric scalar types.
/// For example: Matrix<RGBASample<Float>> -> NHWCTensor<Float>
///
public protocol UniformDenseScalar {
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
public protocol RGBImageSample: UniformDenseScalar {
    var r: ComponentScalar { get set }
    var g: ComponentScalar { get set }
    var b: ComponentScalar { get set }
}

public struct RGBSample<T>: RGBImageSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var r, g, b: T
    public init() { r = T.zero; g = T.zero; b = T.zero }
}

public protocol RGBAImageSample: UniformDenseScalar {
    var r: ComponentScalar { get set }
    var g: ComponentScalar { get set }
    var b: ComponentScalar { get set }
    var a: ComponentScalar { get set }
}

public struct RGBASample<T> : RGBAImageSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var r, g, b, a: T
    public init() { r = T.zero; g = T.zero; b = T.zero; a = T.zero }
}

//==============================================================================
// Audio sample types
public protocol StereoAudioSample: UniformDenseScalar {
    var left: ComponentScalar { get set }
    var right: ComponentScalar { get set }
}

public struct StereoSample<T>: StereoAudioSample
where T: AnyNumeric & AnyFixedSizeScalar {
    public typealias ComponentScalar = T
    public var left, right: T
    public init() { left = T.zero; right = T.zero }
}

//==============================================================================
// ScalarTensorView
public protocol ScalarView: TensorView where
    BoolView == ScalarTensor<Bool>,
    IndexView == ScalarTensor<IndexScalar>{}

public extension ScalarView {
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(_ value: Scalar,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil) {
        
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorData: nil, viewOffset: 0,
                  isShared: false, scalars: nil)
    }
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar>: ScalarView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar?
    public let shape: DataShape
    public var tensorData: TensorData
    public var viewOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorData: TensorData?,
                viewOffset: Int,
                isShared: Bool,
                scalars: [Scalar]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.tensorData = TensorData()
        initTensorData(tensorData, name, scalars)
    }
}

//==============================================================================
// VectorView
public protocol VectorView: TensorView
where BoolView == Vector<Bool>, IndexView == Vector<IndexScalar> {
    // properties
    var count: Int { get }
}

public extension VectorView {
    /// the number of elements in the vector
    var count: Int { return shape.extents[0] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    /// create empty space
    init(count: Int,
         padding: [Padding]? = nil,
         padValue: Scalar? = nil,
         name: String? = nil) {
        
        let shape = DataShape(extents: [count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorData: nil, viewOffset: 0,
                  isShared: false, scalars: nil)
    }
    
    /// initialize with scalar array
    init(name: String? = nil,
         padding: [Padding]? = nil,
         padValue: Scalar? = nil,
         scalars: [Scalar]) {
        let shape = DataShape(extents: [scalars.count])
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorData: nil, viewOffset: 0,
                  isShared: false, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// Vector
public struct Vector<Scalar>: VectorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar?
    public let shape: DataShape
    public var tensorData: TensorData
    public var viewOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorData: TensorData?,
                viewOffset: Int,
                isShared: Bool,
                scalars: [Scalar]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.tensorData = TensorData()
        initTensorData(tensorData, name, scalars)
    }
}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView
where BoolView == Matrix<Bool>, IndexView == Matrix<IndexScalar> {
    // properties
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension MatrixView {
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
    
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil,
         isColMajor: Bool = false, scalars: [Scalar]? = nil) {
        
        let shape = !isColMajor ? DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorData: nil, viewOffset: 0,
                  isShared: false, scalars: scalars)
    }
    
    /// initialize with explicit labels
    init(_ rows: Int, _ cols: Int, isColMajor: Bool = false,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil,
         scalars: [Scalar]? = nil) {
        self.init(extents: [rows, cols],
                  padding: padding, padValue: padValue,
                  name: name, isColMajor: isColMajor, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
// Matrix
public struct Matrix<Scalar>: MatrixView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar?
    public let shape: DataShape
    public var tensorData: TensorData
    public var viewOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorData: TensorData?,
                viewOffset: Int = 0,
                isShared: Bool = false,
                scalars: [Scalar]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.tensorData = TensorData()
        initTensorData(tensorData, name, scalars)
    }
}

//==============================================================================
// VolumeView
public protocol VolumeView: TensorView
where BoolView == Volume<Bool>, IndexView == Volume<IndexScalar> {
    var depthStride: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension VolumeView {
    // properties
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2] }

    //--------------------------------------------------------------------------
    /// shaped initializers
    init(extents: [Int],
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil,
         scalars: [Scalar]? = nil) {
        
        let shape = DataShape(extents: extents)
        self.init(shape: shape, dataShape: shape, name: name,
                  padding: padding, padValue: padValue,
                  tensorData: nil, viewOffset: 0,
                  isShared: false, scalars: scalars)
    }
    
    /// initialize with explicit labels
    init(_ depths: Int, _ rows: Int, _ cols: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil,
         scalars: [Scalar]? = nil) {
        
        self.init(extents: [depths, rows, cols],
                  padding: padding, padValue: padValue,
                  name: name, scalars: scalars)
    }
}

//------------------------------------------------------------------------------
/// Volume
public struct Volume<Scalar>: VolumeView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar?
    public let shape: DataShape
    public var tensorData: TensorData
    public var viewOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorData: TensorData?,
                viewOffset: Int,
                isShared: Bool,
                scalars: [Scalar]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.tensorData = TensorData()
        initTensorData(tensorData, name, scalars)
    }
}

//==============================================================================
// NDTensorView
public protocol NDTensorView: TensorView
where BoolView == NDTensor<Bool>, IndexView == NDTensor<IndexScalar> {
    
}

//------------------------------------------------------------------------------
// NDTensor
// This is an n-dimentional tensor without specialized extent accessors
public struct NDTensor<Scalar>: NDTensorView {
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar?
    public let shape: DataShape
    public var tensorData: TensorData
    public var viewOffset: Int
    
    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Scalar?,
                tensorData: TensorData?,
                viewOffset: Int,
                isShared: Bool,
                scalars: [Scalar]?) {
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue
        self.isShared = isShared
        self.viewOffset = viewOffset
        self.tensorData = TensorData()
        initTensorData(tensorData, name, scalars)
    }
}

////==============================================================================
///// NCHWTensorView
///// An NCHW tensor is a standard layout for use with cuDNN.
///// It has a layout of numerics organized as:
///// n: items
///// c: channels
///// h: rows
///// w: cols
//public protocol NCHWTensorView: TensorView
//where BoolView == NCHWTensor<Bool>, IndexView == NCHWTensor<IndexScalar> {
//    var rowStride: Int { get }
//    var colStride: Int { get }
//}
//
//public extension NCHWTensorView {
//    // properties
//    var itemStride: Int { return shape.strides[0] }
//    var channelStride: Int { return shape.strides[1] }
//    var rowStride: Int { return shape.strides[2] }
//    var colStride: Int { return shape.strides[3] }
//
//    //--------------------------------------------------------------------------
//    /// shaped initializers
//    init(extents: [Int],
//         padding: [Padding]? = nil, padValue: Scalar? = nil,
//         name: String? = nil, logging: LogInfo? = nil,
//         scalars: [Scalar]? = nil) {
//
//        let shape = DataShape(extents: extents)
//        self.init(shape: shape,
//                  padding: padding, padValue: padValue,
//                  name: name, logging: logging)
//
//        // it's being initialized in host memory so it can't fail
//        if let scalars = scalars {
//            _ = try! readWrite().initialize(from: scalars)
//        }
//    }
//
//    /// initialize with explicit labels
//    init(_ items: Int, _ channels: Int, _ rows: Int, _ cols: Int,
//         padding: [Padding]? = nil, padValue: Scalar? = nil,
//         name: String? = nil, logging: LogInfo? = nil,
//         scalars: [Scalar]? = nil) {
//
//        self.init(extents: [items, channels, rows, cols],
//                  padding: padding, padValue: padValue,
//                  name: name, logging: logging, scalars: scalars)
//    }
//}
//
////------------------------------------------------------------------------------
//// NCHWTensor
//public struct NCHWTensor<Scalar>: NCHWTensorView {
//    // properties
//    public var dataShape: DataShape? = nil
//    public var _isReadOnly: Bool = false
//    public var _isShared: Bool = false
//    public var _name: String? = nil
//    public var _lastAccessCopiedTensorData: Bool = false
//    public var _shape: DataShape = DataShape()
//    public var tensorData: TensorData = TensorData()
//    public var viewOffset: Int = 0
//    public var logging: LogInfo? = nil
//    public var padding: [Padding]? = nil
//    public var padValue: Scalar?
//    public init() {}
//}
//
////==============================================================================
///// NHWCTensorView
///// An NHWC tensor is a standard layout for use with cuDNN.
///// It has a layout of numerics organized as:
///// n: items
///// h: rows
///// w: cols
///// c: channels
//public protocol NHWCTensorView: TensorView
//where BoolView == NHWCTensor<Bool>, IndexView == NHWCTensor<IndexScalar> {
//    var rowStride: Int { get }
//    var colStride: Int { get }
//}
//
//public extension NHWCTensorView {
//    // properties
//    var itemStride: Int { return shape.strides[0] }
//    var channelStride: Int { return shape.strides[1] }
//    var rowStride: Int { return shape.strides[2] }
//    var colStride: Int { return shape.strides[3] }
//
//    //--------------------------------------------------------------------------
//    /// shaped initializers
//    init(extents: [Int],
//         padding: [Padding]? = nil, padValue: Scalar? = nil,
//         name: String? = nil, logging: LogInfo? = nil,
//         scalars: [Scalar]? = nil) {
//
//        let shape = DataShape(extents: extents)
//        self.init(shape: shape,
//                  padding: padding, padValue: padValue,
//                  name: name, logging: logging)
//
//        // it's being initialized in host memory so it can't fail
//        if let scalars = scalars {
//            _ = try! readWrite().initialize(from: scalars)
//        }
//    }
//
//    /// initialize with explicit labels
//    init(_ items: Int, _ rows: Int, _ cols: Int, _ channels: Int,
//         padding: [Padding]? = nil, padValue: Scalar? = nil,
//         name: String? = nil, logging: LogInfo? = nil,
//         scalars: [Scalar]? = nil) {
//
//        self.init(extents: [items, rows, cols, channels],
//                  padding: padding, padValue: padValue,
//                  name: name, logging: logging, scalars: scalars)
//    }
//}
//
////------------------------------------------------------------------------------
//// NHWCTensor
//public struct NHWCTensor<Scalar>: NHWCTensorView {
//    // properties
//    public var dataShape: DataShape? = nil
//    public var _isReadOnly: Bool = false
//    public var _isShared: Bool = false
//    public var _name: String? = nil
//    public var _lastAccessCopiedTensorData: Bool = false
//    public var _shape: DataShape = DataShape()
//    public var tensorData: TensorData = TensorData()
//    public var viewOffset: Int = 0
//    public var logging: LogInfo? = nil
//    public var padding: [Padding]? = nil
//    public var padValue: Scalar?
//    public init() {}
//}
//
////------------------------------------------------------------------------------
//public extension NHWCTensor {
//    /// zero copy cast of a matrix of dense uniform scalars to NHWC
//    init<M: MatrixView>(_ matrix: M, name: String? = nil) where
//        M.Scalar: UniformDenseScalar,
//        M.Scalar.ComponentScalar == Scalar {
//            let extents = [1, matrix.shape.extents[0],
//                           matrix.shape.extents[1], M.Scalar.componentCount]
//            self.init(shape: DataShape(extents: extents),
//                      tensorData: matrix.tensorData, viewOffset: 0,
//                      isShared: false, name: nil, logging: matrix.logging)
//    }
//}
