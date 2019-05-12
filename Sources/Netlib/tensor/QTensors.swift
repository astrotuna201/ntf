//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
// Quantizing
public protocol Quantizing where Self: TensorView, Scalar: Quantizable {
    /// the scalar type presented by the view
    associatedtype Viewed: Quantizable
    
    /// the bias to apply during conversion
    var bias: Float { get set }
    /// the scale to apply during conversion
    var scale: Float { get set }
    /// converts Stored <--> Viewed
    var quantizer: Quantizer<Scalar, Viewed> { get set }
}

public extension Quantizing {
    var bias: Float {
        get { return quantizer.bias }
        set { quantizer.bias = newValue }
    }
    
    var scale: Float {
        get { return quantizer.scale }
        set { quantizer.scale = newValue }
    }
}

//==============================================================================
// QMatrix
public struct QMatrix<Scalar, Viewed>: MatrixView, Quantizing where
    Scalar: Quantizable & DefaultInitializer,
    Viewed: Quantizable
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
    public var quantizer: Quantizer<Scalar, Viewed>
    
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
        self.quantizer = Quantizer<Scalar, Viewed>()
        initTensorArray(tensorArray, name, scalars)
    }
}


