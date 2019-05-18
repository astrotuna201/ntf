//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

////==============================================================================
//// ShapedTensorView
//public protocol QShapedTensorView: TensorView { }
//
//public extension QShapedTensorView where  {
//    //--------------------------------------------------------------------------
//    /// DenseView
//    func createDenseView(with value: Values.Element) -> Self {
//        let extents = [Int](repeating: 1, count: rank)
//        let shape = DataShape(extents: extents)
//        return Self(shape: shape, dataShape: shape, name: name,
//                    tensorArray: nil, viewDataOffset: 0,
//                    indexAlignment: nil, isShared: false, values: [value])
//    }
//}

////==============================================================================
//// Indexing
//public extension ShapedTensorView {
//    /// returns a collection of read only values
//    func values(using stream: DeviceStream?) throws
//        -> TensorValueCollection<Self>
//    {
//        let buffer = try readOnly(using: stream)
//        return try TensorValueCollection(view: self, buffer: buffer)
//    }
//
//    /// returns a collection of read write values
//    mutating func mutableValues(using stream: DeviceStream?) throws
//        -> TensorMutableValueCollection<Self>
//    {
//        let buffer = try readWrite(using: stream)
//        return try TensorMutableValueCollection(view: &self, buffer: buffer)
//    }
//}

//==============================================================================
// QMatrix
public struct QMatrix<Element, Viewed>: MatrixView, Quantizing where
    Element: Quantizable, Viewed: Quantizable
{
    public func createDenseView(with value: Viewed) -> QMatrix {
        let value = convert(viewed: value)
        let extents = [Int](repeating: 1, count: rank)
        let shape = DataShape(extents: extents)
        return QMatrix(shape: shape, dataShape: shape, name: name,
                       tensorArray: nil, viewDataOffset: 0,
                       indexAlignment: nil, isShared: false,
                       values: [value])
    }
    
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
    
    /// returns a collection of read only values
    public func values(using stream: DeviceStream?) throws
        -> QTensorValueCollection<QMatrix>
    {
        let buffer = try readOnly(using: stream)
        return try QTensorValueCollection(view: self, buffer: buffer)
    }
    
    /// returns a collection of read write values
    public mutating func mutableValues(using stream: DeviceStream?) throws
        -> QTensorMutableValueCollection<QMatrix>
    {
        let buffer = try readWrite(using: stream)
        return try QTensorMutableValueCollection(view: &self, buffer: buffer)
    }
}
