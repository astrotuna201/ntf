//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// QShapedTensorView
public protocol QShapedTensorView: Quantizing {
    /// fully specified used for creating views
    init(shape: DataShape,
         dataShape: DataShape,
         name: String?,
         padding: [Padding]?,
         padValue: Element?,
         tensorArray: TensorArray?,
         viewDataOffset: Int,
         isShared: Bool,
         quantizer: Q,
         scalars: [Element]?)
}

public extension QShapedTensorView {
    //--------------------------------------------------------------------------
    /// empty
    init() {
        fatalError()
//        self.init(shape: DataShape(),
//                  dataShape: DataShape(),
//                  name: nil,
//                  padding: nil,
//                  padValue: nil,
//                  tensorArray: TensorArray(),
//                  viewDataOffset: 0,
//                  isShared: false,
//                  quantizer: QuantizeVoid<Element, Viewed>(),
//                  scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// init(with extents:
    /// convenience initializer used by generics to create typed result
    /// views of matching shape
    init(with extents: [Int], quantizer: Q) {
        let shape = DataShape(extents: extents)
        self.init(shape: shape,
                  dataShape: shape,
                  name: nil,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false,
                  quantizer: quantizer,
                  scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// init(value:
    /// convenience initializer used by generics
    /// - Parameter value: the initial value to set
    init(with value: Element, quantizer: Q) {
        // create scalar version of the shaped view type
        let shape = DataShape(extents: [1])
        self.init(shape: shape, dataShape: shape, name: nil,
                  padding: nil, padValue: nil,
                  tensorArray: nil, viewDataOffset: 0,
                  isShared: false,
                  quantizer: quantizer,
                  scalars: [value])
    }
    
    //--------------------------------------------------------------------------
    /// repeated view
    init(with extents: [Int], repeating other: Self, quantizer: Q) {
        self.init(shape: DataShape(extents: extents),
                  dataShape: other.shape,
                  name: other.name,
                  padding: nil,
                  padValue: other.padValue,
                  tensorArray: other.tensorArray,
                  viewDataOffset: other.viewDataOffset,
                  isShared: other.isShared,
                  quantizer: quantizer,
                  scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    // realizing init
    init<T>(realizing other: T, quantizer: Q) throws where
        T: TensorView, Viewed == T.Viewed
    {
        if let other = other as? Self,
            other.isContiguous && other.traversal == .normal
        {
            self = other
        } else {
            var result = Self(with: other.extents, quantizer: quantizer)
            Netlib.copy(view: other, result: &result)
            self = result
        }
    }
    
    //--------------------------------------------------------------------------
    /// createSubView
    /// Returns a view of the tensorArray relative to this view
    func createView(at offset: [Int], with extents: [Int],
                    isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))
        
        // the subview offset is the current plus the offset of index
        let subViewOffset = viewDataOffset + shape.linearIndex(of: offset)
        let subViewShape = DataShape(extents: extents, strides: shape.strides)
        let name = "\(self.name).subview"
        
        return Self(shape: subViewShape,
                    dataShape: subViewShape,
                    name: name,
                    padding: padding,
                    padValue: padValue,
                    tensorArray: tensorArray,
                    viewDataOffset: subViewOffset,
                    isShared: isReference,
                    quantizer: quantizer,
                    scalars: nil)
    }
    
    //--------------------------------------------------------------------------
    /// reference
    /// creation of a reference is for the purpose of reshaped writes
    /// and multi-threaded writes to prevent mutation.
    /// The data will be copied before reference view creation if
    /// not uniquely held. Reference views will not perform
    /// copy-on-write when a write pointer is taken
    mutating func reference(using stream: DeviceStream) throws -> Self {
        // get the queue, if we reference it as a tensorArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return try queue.sync {
            try copyIfMutates(using: stream)
            return Self(shape: shape,
                        dataShape: dataShape,
                        name: name,
                        padding: padding,
                        padValue: padValue,
                        tensorArray: tensorArray,
                        viewDataOffset: viewDataOffset,
                        isShared: true,
                        quantizer: quantizer,
                        scalars: nil)
        }
    }
    
    //--------------------------------------------------------------------------
    /// flattened
    /// Returns a view with all dimensions higher than `axis` set to 1
    /// and the extent of `axis` adjusted to be the new total element count
    func flattened(axis: Int = 0) -> Self {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }
        
        // create flattened view
        let flatShape = shape.flattened()
        return Self(shape: flatShape,
                    dataShape: flatShape,
                    name: name,
                    padding: padding,
                    padValue: padValue,
                    tensorArray: tensorArray,
                    viewDataOffset: viewDataOffset,
                    isShared: isShared,
                    quantizer: quantizer,
                    scalars: nil)
    }
}

//==============================================================================
// Indexing
public extension QShapedTensorView {
    
    func values(using stream: DeviceStream?) throws -> QTensorValueCollection<Self> {
        let buffer = try readOnly(using: stream)
        return try QTensorValueCollection(view: self, buffer: buffer)
    }
    
    mutating func mutableValues(using stream: DeviceStream?) throws
        -> QTensorMutableValueCollection<Self>
    {
        let buffer = try readWrite(using: stream)
        return try QTensorMutableValueCollection(view: &self, buffer: buffer)
    }
}

//==============================================================================
/// QVectorView
public protocol QVectorView: QShapedTensorView {
    
}

////==============================================================================
///// Vector
//public struct QVector<Element, Viewed, Q>: QVectorView where
//    Element: DefaultInitializer,
//    Q: Quantizer,
//    Q.Element == Element,
//    Q.Viewed == Viewed
//{
//    // properties
//    public let dataShape: DataShape
//    public let isShared: Bool
//    public let padding: [Padding]?
//    public let padValue: Element
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
//                padValue: Element?,
//                tensorArray: TensorArray?,
//                viewDataOffset: Int,
//                isShared: Bool,
//                quantizer: Q,
//                scalars: [Element]?) {
//
//        assert(scalars == nil || scalars!.count == shape.elementCount,
//               "tensor size and scalars count do not match")
//        self.shape = shape
//        self.dataShape = dataShape
//        self.padding = padding
//        self.padValue = padValue ?? Element()
//        self.traversal = initTraversal(padding, shape != dataShape)
//        self.isShared = isShared
//        self.viewDataOffset = viewDataOffset
//        self.quantizer = quantizer
//        self.tensorArray = TensorArray()
//        initTensorArray(tensorArray, name, scalars)
//    }
//}
