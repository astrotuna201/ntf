//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// Quantizing TensorView common extensions
public extension TensorView where Self: Quantizing, Values.Element == Viewed {
    //--------------------------------------------------------------------------
    /// DenseView
    func createDenseView(_ value: Values.Element, name: String? = nil) -> Self {
        let extents = [Int](repeating: 1, count: rank)
        let shape = DataShape(extents: extents)
        let elements = [convert(viewed: value)]
        let array = try! TensorArray(copying: elements,
                                     name: name ?? String(describing: Self.self),
                                     using: _Streams.hostStream)
        return Self(shape: shape, dataShape: shape,
                    tensorArray: array, viewDataOffset: 0,
                    indexAlignment: zeroAlignment(shape.rank),
                    traversal: .normal, isShared: false)
    }
}

//==============================================================================
// MatrixView data initialization extensions
public extension MatrixView where Self: Quantizing, Values.Element == Viewed {
    //-------------------------------------
    /// with single value
    init(_ value: Viewed, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        self.init(shape: shape, dataShape: shape,
                  tensorArray: TensorArray(),
                  viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)

        let element = convert(viewed: value)
        tensorArray =
            try! TensorArray(copying: [element],
                             name: name ?? String(describing: Self.self),
                             using: _Streams.hostStream)
    }
    
    //-------------------------------------
    /// with convertable collection
    /// TODO: should the collection be lazy??
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, any: C) where
        C: Collection, C.Element: AnyConvertable, Viewed: AnyConvertable
    {
        self.init(extents, name: name, layout: layout,
                  values: any.map { Viewed(any: $0) })
    }
    
    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, values: C) where
        C: Collection, C.Element == Viewed
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        
        self.init(shape: shape, dataShape: shape,
                  tensorArray: TensorArray(), viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)

        tensorArray =
            try! TensorArray(copying: values.map { convert(viewed: $0) },
                             name: name ?? String(describing: Self.self),
                             using: _Streams.hostStream)
    }
}

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
                tensorArray: TensorArray,
                viewDataOffset: Int,
                indexAlignment: [Int],
                traversal: TensorTraversal,
                isShared: Bool)
    {
        self.shape = shape
        self.dataShape = dataShape
        self.tensorArray = tensorArray
        self.viewDataOffset = viewDataOffset
        self.indexAlignment = indexAlignment
        self.isShared = isShared
        self.traversal = traversal
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
