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
    IndexView == ScalarTensor<TensorIndex>{}

public extension ScalarView {
    //--------------------------------------------------------------------------
    /// shaped initializers
    init(_ value: Scalar,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil) {
        
        let shape = DataShape(extents: [1])
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name)
    }
}

//------------------------------------------------------------------------------
// ScalarTensor
public struct ScalarTensor<Scalar>: ScalarView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessCopiedTensorData: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var padding: [Padding]? = nil
    public var padValue: Scalar? = nil
    public init() {}
}

//==============================================================================
// VectorView
public protocol VectorView: TensorView
where BoolView == Vector<Bool>, IndexView == Vector<TensorIndex> {
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
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil) {
        
        let shape = DataShape(extents: [count])
        self.init(shape: shape,
                  padding: padding, padValue: padValue,
                  name: name)
    }
    
    /// initialize with scalar array
    init(name: String? = nil,
         padding: [Padding]? = nil, padValue: Scalar? = nil, scalars: [Scalar]) {
        
        self.init(count: scalars.count,
                  padding: padding, padValue: padValue,
                  name: name)
        _ = try! readWrite().initialize(from: scalars)
    }
}

//------------------------------------------------------------------------------
// Vector
public struct Vector<Scalar>: VectorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessCopiedTensorData: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var padding: [Padding]? = nil
    public var padValue: Scalar?
    public init() {}
}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView
where BoolView == Matrix<Bool>, IndexView == Matrix<TensorIndex> {
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
        
        let shape = DataShape(extents: extents)
        self.init(shape: shape, padding: padding, padValue: padValue,name: name)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
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
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessCopiedTensorData: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var padding: [Padding]? = nil
    public var padValue: Scalar?
    public init() {}
}

//==============================================================================
// VolumeView
public protocol VolumeView: TensorView
where BoolView == Volume<Bool>, IndexView == Volume<TensorIndex> {
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
        
        self.init(shape: DataShape(extents: extents),
                  padding: padding, padValue: padValue,
                  name: name)
        
        // it's being initialized in host memory so it can't fail
        if let scalars = scalars {
            _ = try! readWrite().initialize(from: scalars)
        }
    }
    
    /// initialize with explicit labels
    init(_ depths: Int, _ rows: Int, _ cols: Int,
         padding: [Padding]? = nil, padValue: Scalar? = nil,
         name: String? = nil,
         scalars: [Scalar]? = nil) {
        
        self.init(extents: [depths, rows, cols],
                  padding: padding, padValue: padValue,
                  name: name,
                  scalars: scalars)
    }
}

//------------------------------------------------------------------------------
/// Volume
public struct Volume<Scalar>: VolumeView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessCopiedTensorData: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var padding: [Padding]? = nil
    public var padValue: Scalar?
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
public struct NDTensor<Scalar>: NDTensorView {
    // properties
    public var _dataShape: DataShape? = nil
    public var _isReadOnly: Bool = false
    public var _isShared: Bool = false
    public var _name: String? = nil
    public var _lastAccessCopiedTensorData: Bool = false
    public var _shape: DataShape = DataShape()
    public var _tensorData: TensorData = TensorData()
    public var _viewOffset: Int = 0
    public var padding: [Padding]? = nil
    public var padValue: Scalar?
    public init() {}
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
//where BoolView == NCHWTensor<Bool>, IndexView == NCHWTensor<TensorIndex> {
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
//    public var _dataShape: DataShape? = nil
//    public var _isReadOnly: Bool = false
//    public var _isShared: Bool = false
//    public var _name: String? = nil
//    public var _lastAccessCopiedTensorData: Bool = false
//    public var _shape: DataShape = DataShape()
//    public var _tensorData: TensorData = TensorData()
//    public var _viewOffset: Int = 0
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
//where BoolView == NHWCTensor<Bool>, IndexView == NHWCTensor<TensorIndex> {
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
//    public var _dataShape: DataShape? = nil
//    public var _isReadOnly: Bool = false
//    public var _isShared: Bool = false
//    public var _name: String? = nil
//    public var _lastAccessCopiedTensorData: Bool = false
//    public var _shape: DataShape = DataShape()
//    public var _tensorData: TensorData = TensorData()
//    public var _viewOffset: Int = 0
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
//                      tensorData: matrix._tensorData, viewOffset: 0,
//                      isShared: false, name: nil, logging: matrix.logging)
//    }
//}
