//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// Quantizing
public protocol Quantizing where Self: TensorView, Scalar: FixedWidthInteger {
    /// the scalar type presented by the view
    associatedtype Viewed: BinaryFloatingPoint
    
    /// the bias to apply during conversion
    var bias: Float { get set }
    /// the scale to apply during conversion
    var scale: Float { get set }
    /// the scale factor used to map into the range of the type, times
    /// the user scale factor
    var _transformScale: Float { get set }
    /// a private scale factor used by the transform functions
    var _inverseTransformScale: Float { get set }
    
    //--------------------------------------------------------------------------
    /// converts from Scalar to ViewedScalar
    func convert(stored: Scalar) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Scalar
}

public extension Quantizing {
    mutating func updateScales() {
        _transformScale = (Float(Scalar.max) + 1) * scale
        _inverseTransformScale = 1 / _transformScale
    }
    
    @inlinable @inline(__always)
    func convert(stored: Scalar) -> Viewed {
        if stored == 0 {
            return Viewed(bias)
        } else if stored > 0 {
            return Viewed((Float(stored) + 1) * _inverseTransformScale + bias)
        } else {
            return Viewed((Float(stored)) * _inverseTransformScale + bias)
        }
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Scalar {
        if viewed == Viewed(bias) {
            return 0
        } else if viewed > 0 {
            return Scalar(((Float(viewed) - bias) * _transformScale) - 1)
        } else {
            return Scalar((Float(viewed) - bias) * _transformScale)
        }
    }
}

//==============================================================================
// QMatrix
public struct QMatrix<Scalar, Viewed>: MatrixView, Quantizing where
    Scalar: FixedWidthInteger & DefaultInitializer,
    Viewed: BinaryFloatingPoint
{
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Scalar
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    // quant
    public var bias: Float = 0
    public var scale: Float = 1
    public var _transformScale: Float = 1
    public var _inverseTransformScale: Float = 1

    // initializer
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
        initTensorArray(tensorArray, name, scalars)
        updateScales()
    }
}


