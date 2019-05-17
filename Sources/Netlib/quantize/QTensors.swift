//******************************************************************************
//  Created by Edward Connell on 5/1/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

////==============================================================================
//// QShapedTensorView
//public protocol QShapedTensorView: Quantizing {
//    /// fully specified used for creating views
//    init(shape: DataShape,
//         dataShape: DataShape,
//         name: String?,
//         padding: [Padding]?,
//         padValue: Element?,
//         tensorArray: TensorArray?,
//         viewDataOffset: Int,
//         isShared: Bool,
//         quantizer: Quant,
//         values: [Values.Element]?)
//}
//
//public extension QShapedTensorView {
//    //--------------------------------------------------------------------------
//    /// DenseView
//    func createDenseView(with extents: [Int],
//                         values: [Values.Element]? = nil) -> Self
//    {
//        let shape = DataShape(extents: extents)
//        return Self(
//            shape: shape, dataShape: shape, name: name,
//            padding: nil, padValue: nil,
//            tensorArray: nil, viewDataOffset: 0,
//            isShared: false,
//            quantizer: quantizer,
//            values: values)
//    }
//
//    //--------------------------------------------------------------------------
//    /// repeated view
//    init(with extents: [Int], repeating other: Self) {
//        self.init(shape: DataShape(extents: extents),
//                  dataShape: other.shape,
//                  name: other.name,
//                  padding: nil,
//                  padValue: other.padValue,
//                  tensorArray: other.tensorArray,
//                  viewDataOffset: other.viewDataOffset,
//                  isShared: other.isShared,
//                  quantizer: other.quantizer,
//                  values: nil)
//    }
//
//    //--------------------------------------------------------------------------
//    /// createSubView
//    /// Returns a view of the tensorArray relative to this view
//    func createView(at offset: [Int], with extents: [Int],
//                    isReference: Bool) -> Self {
//        // validate
//        assert(offset.count == shape.rank && extents.count == shape.rank)
//        assert(shape.contains(offset: offset, extents: extents))
//
//        // the subview offset is the current plus the offset of index
//        let subViewOffset = viewDataOffset + shape.linearIndex(of: offset)
//        let subViewShape = DataShape(extents: extents, strides: shape.strides)
//        let name = "\(self.name).subview"
//
//        return Self(shape: subViewShape,
//                    dataShape: subViewShape,
//                    name: name,
//                    padding: padding,
//                    padValue: padValue,
//                    tensorArray: tensorArray,
//                    viewDataOffset: subViewOffset,
//                    isShared: isReference,
//                    quantizer: quantizer,
//                    values: nil)
//    }
//
//    //--------------------------------------------------------------------------
//    /// reference
//    /// creation of a reference is for the purpose of reshaped writes
//    /// and multi-threaded writes to prevent mutation.
//    /// The data will be copied before reference view creation if
//    /// not uniquely held. Reference views will not perform
//    /// copy-on-write when a write pointer is taken
//    mutating func reference(using stream: DeviceStream) throws -> Self {
//        // get the queue, if we reference it as a tensorArray member it
//        // it adds a ref count which messes things up
//        let queue = tensorArray.accessQueue
//
//        return try queue.sync {
//            try copyIfMutates(using: stream)
//            return Self(shape: shape,
//                        dataShape: dataShape,
//                        name: name,
//                        padding: padding,
//                        padValue: padValue,
//                        tensorArray: tensorArray,
//                        viewDataOffset: viewDataOffset,
//                        isShared: true,
//                        quantizer: quantizer,
//                        values: nil)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    /// flattened
//    /// Returns a view with all dimensions higher than `axis` set to 1
//    /// and the extent of `axis` adjusted to be the new total element count
//    func flattened(axis: Int = 0) -> Self {
//        // check if self already meets requirements
//        guard self.isShared != isShared || axis != shape.rank - 1 else {
//            return self
//        }
//
//        // create flattened view
//        let flatShape = shape.flattened()
//        return Self(shape: flatShape,
//                    dataShape: flatShape,
//                    name: name,
//                    padding: padding,
//                    padValue: padValue,
//                    tensorArray: tensorArray,
//                    viewDataOffset: viewDataOffset,
//                    isShared: isShared,
//                    quantizer: quantizer,
//                    values: nil)
//    }
//
//    //--------------------------------------------------------------------------
//    /// initTensorArray
//    /// a helper to correctly initialize the tensorArray object
//    mutating func initTensorArray(_ tensorData: TensorArray?,
//                                  _ name: String?,
//                                  _ quantizer: Quant,
//                                  _ values: [Values.Element]?) {
//        if let tensorData = tensorData {
//            tensorArray = tensorData
//        } else {
//            assert(shape.isContiguous, "new views should have a dense shape")
//            let name = name ?? String(describing: Self.self)
//
//            // allocate backing tensorArray
//            if let values = values {
//                assert(values.count == dataShape.elementCount,
//                       "number of values does not match tensor extents")
//                do {
//                    let elements = values.map { quantizer.convert(viewed: $0) }
//                    tensorArray = try elements.withUnsafeBufferPointer {
//                        try TensorArray(copying: $0, name: name)
//                    }
//                } catch {
//                    tensorArray = TensorArray()
//                    _Streams.current.reportDevice(error: error)
//                }
//            } else {
//                tensorArray = TensorArray(type: Element.self,
//                                          count: dataShape.elementCount,
//                                          name: name)
//            }
//        }
//    }
//}
//
////==============================================================================
//// Indexing
//public extension QShapedTensorView {
//
//    func values(using stream: DeviceStream?) throws -> QTensorValueCollection<Self> {
//        let buffer = try readOnly(using: stream)
//        return try QTensorValueCollection(view: self, buffer: buffer)
//    }
//
//    mutating func mutableValues(using stream: DeviceStream?) throws
//        -> QTensorMutableValueCollection<Self>
//    {
//        let buffer = try readWrite(using: stream)
//        return try QTensorMutableValueCollection(view: &self, buffer: buffer)
//    }
//}
//
////==============================================================================
///// QVectorView
//public protocol QVectorView: QShapedTensorView { }
//
////extension QVector: CustomStringConvertible where Element: AnyConvertable {
////    public var description: String { return formatted() }
////}
//
////==============================================================================
//// QVectorView extensions
//public extension QVectorView {
//    //--------------------------------------------------------------------------
//    var endIndex: VectorIndex {
//        return VectorIndex(endOf: self)
//    }
//
//    var startIndex: VectorIndex {
//        return VectorIndex(view: self, at: 0)
//    }
//
//    //--------------------------------------------------------------------------
//    /// shaped initializers
//    /// with single value
//    init(_ value: Values.Element, quantizer: Quant,
//         name: String? = nil) {
//        let shape = DataShape(extents: [1])
//        self.init(shape: shape, dataShape: shape, name: name,
//                  padding: nil, padValue: nil,
//                  tensorArray: nil, viewDataOffset: 0,
//                  isShared: false, quantizer: quantizer, values: [value])
//    }
//
//    //-------------------------------------
//    /// empty array
//    init(count: Int, quantizer: Quant, name: String? = nil) {
//        let shape = DataShape(extents: [count])
//        self.init(shape: shape, dataShape: shape, name: name,
//                  padding: nil, padValue: nil,
//                  tensorArray: nil, viewDataOffset: 0,
//                  isShared: false, quantizer: quantizer, values: nil)
//    }
//
//    //-------------------------------------
//    /// with Array
//    init(name: String? = nil, quantizer: Quant, values: [Values.Element]) {
//        let shape = DataShape(extents: [values.count])
//        self.init(shape: shape, dataShape: shape, name: name,
//                  padding: nil, padValue: nil,
//                  tensorArray: nil, viewDataOffset: 0,
//                  isShared: false, quantizer: quantizer, values: values)
//    }
//
//    //-------------------------------------
//    /// with Sequence
//    init<Seq>(name: String? = nil, quantizer: Quant, sequence: Seq) where
//        Seq: Sequence, Seq.Element: AnyConvertable,
//        Values.Element: AnyConvertable
//    {
//        let values = Self.sequence2ScalarArray(sequence)
//        let shape = DataShape(extents: [values.count])
//        self.init(shape: shape, dataShape: shape, name: name,
//                  padding: nil, padValue: nil,
//                  tensorArray: nil, viewDataOffset: 0,
//                  isShared: false, quantizer: quantizer, values: values)
//    }
//
//    //-------------------------------------
//    /// with reference to read only buffer
//    /// useful for memory mapped databases, or hardware device buffers
//    init(referenceTo buffer: UnsafeBufferPointer<Element>,
//         quantizer: Quant, name: String? = nil)
//    {
//        // create tensor data reference to buffer
//        let name = name ?? String(describing: Self.self)
//        let tensorArray = TensorArray(referenceTo: buffer, name: name)
//
//        // create shape considering column major
//        let shape = DataShape(extents: [buffer.count])
//        self.init(shape: shape, dataShape: shape, name: name,
//                  padding: nil, padValue: nil,
//                  tensorArray: tensorArray, viewDataOffset: 0,
//                  isShared: false, quantizer: quantizer, values: nil)
//    }
//
//    //--------------------------------------------------------------------------
//    /// BoolView
//    func createBoolView(with extents: [Int]) -> Vector<Bool> {
//        let shape = DataShape(extents: extents)
//        return Vector<Bool>(
//            shape: shape, dataShape: shape, name: name,
//            padding: nil, padValue: nil,
//            tensorArray: nil, viewDataOffset: 0,
//            isShared: false, values: nil)
//    }
//
//    //--------------------------------------------------------------------------
//    /// IndexView
//    func createIndexView(with extents: [Int], values: [IndexElement]? = nil)
//        -> Vector<IndexElement>
//    {
//        let shape = DataShape(extents: extents)
//        return Vector<IndexElement>(
//            shape: shape, dataShape: shape, name: name,
//            padding: nil, padValue: nil,
//            tensorArray: nil, viewDataOffset: 0,
//            isShared: false, values: values)
//    }
//}

//==============================================================================
/// Vector
public struct QVector<Element>: VectorView where
    Element: DefaultInitializer
{
    // properties
    public let dataShape: DataShape
    public let isShared: Bool
    public let padding: [Padding]?
    public let padValue: Element
    public let shape: DataShape
    public var tensorArray: TensorArray
    public let traversal: TensorTraversal
    public var viewDataOffset: Int

    public init(shape: DataShape,
                dataShape: DataShape,
                name: String?,
                padding: [Padding]?,
                padValue: Element?,
                tensorArray: TensorArray?,
                viewDataOffset: Int,
                isShared: Bool,
                values: [Values.Element]?) {

        assert(values == nil || values!.count == shape.elementCount,
               "tensor size and scalars count do not match")
        self.shape = shape
        self.dataShape = dataShape
        self.padding = padding
        self.padValue = padValue ?? Element()
        self.traversal = initTraversal(padding, shape != dataShape)
        self.isShared = isShared
        self.viewDataOffset = viewDataOffset
        self.tensorArray = TensorArray()
        initTensorArray(tensorArray, name, values)
    }
}
