//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// QMatrix
public struct QMatrix<Element, Viewed>: MatrixView, Quantizing where
    Element: Quantizable, Viewed: Quantizable
{
    // properties
    public let dataShape: DataShape
    public let indexAlignment: [Int]
    public let isShared: Bool
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int
    public var bias = Viewed(value: Float(0))
    public var scale = Viewed(value: Float(1))

    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                indexAlignment: [Int]?,
                isShared: Bool,
                values: [Element]?) {
        
        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and values count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.traversal = shape == dataShape ? .normal : .repeated
        self.indexAlignment = indexAlignment ?? zeroIndexAlignment(shape.rank)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}
