////******************************************************************************
////  Created by Edward Connell on 5/1/19
////  Copyright © 2019 Connell Research. All rights reserved.
////
//import Foundation
//
////==============================================================================
///// Vector
//public struct QVector<Stored, Viewed, Q>: VectorView, Quantizing where
//    Stored: DefaultInitializer,
//    Q: Quantizer,
//    Q.Stored == Stored,
//    Q.Viewed == Viewed
//{
//    // properties
//    public let dataShape: DataShape
//    public let isShared: Bool
//    public let padding: [Padding]?
//    public let padValue: Stored
//    public let shape: DataShape
//    public var tensorArray: TensorArray
//    public let traversal: TensorTraversal
//    public var viewDataOffset: Int
//    public let quantizer: Q
//
//    public init(shape: DataShape,
//                dataShape: DataShape,
//                name: String?,
//                padding: [Padding]?,
//                padValue: Stored?,
//                tensorArray: TensorArray?,
//                viewDataOffset: Int,
//                isShared: Bool,
//                quantizer: Q,
//                scalars: [Stored]?) {
//
//        assert(scalars == nil || scalars!.count == shape.elementCount,
//               "tensor size and scalars count do not match")
//        self.shape = shape
//        self.dataShape = dataShape
//        self.padding = padding
//        self.padValue = padValue ?? Stored()
//        self.traversal = initTraversal(padding, shape != dataShape)
//        self.isShared = isShared
//        self.viewDataOffset = viewDataOffset
//        self.quantizer = quantizer
//        self.tensorArray = TensorArray()
//        initTensorArray(tensorArray, name, scalars)
//    }
//}
