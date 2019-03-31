//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// TensorDataView
public protocol TensorDataView {
    // types
    associatedtype Scalar: AnyScalar

    //--------------------------------------------------------------------------
    // properties
    var lastAccessMutated: Bool { get }
    var logging: LogInfo? { get set }
    var name: String { get set }
    var shape: DataShape { get }
    var viewOffset: Int { get }
}

public extension TensorDataView where Self: _TensorDataViewImpl {
    var name: String {
        get { return tensorData.name }
        set { tensorData.name = newValue }
    }
}

//==============================================================================
// TensorDataView
public protocol _TensorDataViewImpl {
    // types
    associatedtype Scalar: AnyScalar
    
    //--------------------------------------------------------------------------
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


public extension _TensorDataViewImpl {
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

public extension ScalarTensorView {
}

//==============================================================================
// VectorTensorView
public protocol VectorTensorView: TensorDataView {
    var count: Int { get }
}

public extension VectorTensorView {
    var count: Int { return shape.extents[0] }
}

//==============================================================================
// MatrixTensorView
public protocol MatrixTensorView: TensorDataView {
    var rows: Int { get }
    var cols: Int { get }
    var rowStride: Int { get }
    var colStride: Int { get }
}

public extension MatrixTensorView {
    var rows: Int { return shape.extents[0] }
    var cols: Int { return shape.extents[1] }
    var rowStride: Int { return shape.strides[0] }
    var colStride: Int { return shape.strides[1]  }
}

//==============================================================================
// VolumeTensorView
public protocol VolumeTensorView: MatrixTensorView {
    var depths: Int { get }
    var depthStride: Int { get }
}

public extension VolumeTensorView {
    var depths: Int { return shape.extents[0] }
    var rows: Int { return shape.extents[1] }
    var cols: Int { return shape.extents[2] }
    var depthStride: Int { return shape.strides[0] }
    var rowStride: Int { return shape.strides[1] }
    var colStride: Int { return shape.strides[2]  }
}

//==============================================================================
// Non numeric Scalar types for proof of concept
public struct RGB: AnyScalar {
    var r: UInt8
    var g: UInt8
    var b: UInt8
    public init() { r = 0; g = 0; b = 0 }
}

public struct RGBA: AnyScalar {
    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8
    public init() { r = 0; g = 0; b = 0; a = 0 }
}

//==============================================================================
// NDTensor
public struct NDTensor {
    
}

//==============================================================================
// ScalarTensor
public struct ScalarTensor {
    
}

//==============================================================================
// VectorTensor
public struct VectorTensor {
    
}

//==============================================================================
// StringTensor
public struct StringTensor {
    
}

//==============================================================================
// MatrixTensor
public struct MatrixTensor<Scalar>: TensorDataView, _TensorDataViewImpl
where Scalar: AnyScalar {
    public init(rows: Int, cols: Int,
                name: String? = nil, logging: LogInfo? = nil) {

        self.shape = DataShape(extents: [rows, cols])
        self.logging = logging
        tensorData = TensorData(elementCount: shape.elementCount,
                                 logging: logging, name: name)
    }

    //--------------------------------------------------------------------------
    // properties
    public var isShared: Bool = false
    public var lastAccessMutated: Bool = false
    public var logging: LogInfo?
    public var shape: DataShape
    public var viewOffset: Int = 0
    public var tensorData: TensorData<Scalar>
    
}

//==============================================================================
// RGBImageTensor
public struct RGBImageTensor {
    
}
