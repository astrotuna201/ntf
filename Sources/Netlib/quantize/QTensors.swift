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
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(type: Element.self, count: 1, name: name)
        var view = Self(shape: shape, dataShape: shape,
                        tensorArray: array, viewDataOffset: 0,
                        indexAlignment: zeroAlignment(shape.rank),
                        traversal: .normal, isShared: false)
        try! view.readWrite()[0] = view.convert(viewed: value)
        return view
    }
}

//==============================================================================
// MatrixView data initialization extensions
public extension MatrixView where Self: Quantizing, Values.Element == Viewed {
    //-------------------------------------
    /// with single value
    init(_ value: Viewed, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(type: Element.self, count: 1, name: name)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
        try! readWrite()[0] = convert(viewed: value)
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
        let name = name ?? String(describing: Self.self)
        let array = TensorArray(type: Element.self,
                                count: shape.elementCount, name: name)
        self.init(shape: shape, dataShape: shape,
                  tensorArray: array, viewDataOffset: 0,
                  indexAlignment: zeroAlignment(shape.rank),
                  traversal: .normal, isShared: false)
        // store values
        let buffer = try! readWrite()
        for i in zip(buffer.indices, values.indices) {
            buffer[i.0] = convert(viewed: values[i.1])
        }
    }
    
    func elementArray() throws -> [Element] {
        return [Element](try elementValues())
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
}
